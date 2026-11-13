import logging
import os
from datetime import datetime
from typing import Any
import torch
from sklearn.model_selection import ParameterGrid

from src.core.Callbacks.SPCallbackPyTorch import SPCallbackPyTorch
from src.core.Datasets.SplitManifest import SplitManifest
from src.core.Interfaces.BaseTrainer import BaseTrainer
from src.core.Trainers.FoldTrainer import FoldTrainer
from src.core.Trainers.ResultsRecorder import ResultsRecorder
from src.core.Trainers.TrainingFactory import TrainingFactory
from src.Parser.NeuralRingerTrainerConfiguration import NeuralRingerTrainerConfiguration
from src.utils import create_folder, get_et_eta, get_instance, verify_results

log = logging.getLogger()


class NeuralRingerTrainer(BaseTrainer):
    """Orchestrator for the Neural Ringer training workflow across kinematic regions and folds.

    Attributes:
        config: Configuration specification object.
        use_cuda: Indicates whether CUDA hardware is available.
        device: Active PyTorch computation device.
        factory: TrainingFactory generating models, optimizers, and dataloaders.
        recorder: ResultsRecorder tracking history and serializing checkpoints.
        kfold: Cross-validation splitter instance.
        et: Current transverse energy bin index.
        eta: Current pseudorapidity bin index.
        full_dataset: EgammaNpzDataset containing regional data files.
        results_folder_path: Directory path where output checkpoints are stored.
    """

    def __init__(self, config: NeuralRingerTrainerConfiguration) -> None:
        """Initializes NeuralRingerTrainer with configuration and setup components.

        Args:
            config: NeuralRingerTrainerConfiguration instance.
        """
        self.config: NeuralRingerTrainerConfiguration = config
        self._apply_debug_overrides()
        self.use_cuda: bool = torch.cuda.is_available()
        self._device: torch.device = torch.device("cuda" if self.use_cuda else "cpu")
        self.factory: TrainingFactory = TrainingFactory(self.config, self.device)
        self.recorder: ResultsRecorder = ResultsRecorder()
        self.kfold: Any = self.factory.create_cross_validation()
        self.et: int = -1
        self.eta: int = -1
        self._full_dataset: Any = get_instance(self.config.dataset)
        self._full_dataset.config = config
        self.results_folder_path: str = ""
        self._split_manifest: SplitManifest | None = None

    def _apply_debug_overrides(self) -> None:
        if self.config.debug:
            log.warning("#### EXECUTING ON DEBUG MODE. ####")
            self.config.epochs = 2
            self.config.n_initializations = 1

    def _get_expected_cv_folds(self) -> int:
        """Returns the number of cross-validation folds reserved after holdout split."""
        return max(1, int(self.kfold.get_n_splits()) - 1)

    def _get_missing_repeats_for_fold(
        self, et: int, eta: int, fold_idx: int
    ) -> list[int]:
        """Identifies repeat indices without persisted checkpoint results for a fold.

        Args:
            et: Transverse energy bin index.
            eta: Pseudorapidity bin index.
            fold_idx: Cross-validation fold index.

        Returns:
            List of missing repeat indices.
        """
        return [
            repeat
            for repeat in range(self.config.n_initializations)
            if not verify_results(
                self.results_folder_path, et, eta, repeat, fold_idx
            )
        ]

    def _is_region_complete(self, et: int, eta: int) -> bool:
        """Checks whether all CV fold and repeat checkpoints exist for a region.

        Args:
            et: Transverse energy bin index.
            eta: Pseudorapidity bin index.

        Returns:
            True if all expected checkpoints exist, False otherwise.
        """
        total_folds = self._get_expected_cv_folds()
        for fold_idx in range(total_folds):
            for repeat in range(self.config.n_initializations):
                if not verify_results(
                    self.results_folder_path, et, eta, repeat, fold_idx
                ):
                    return False
        log.info(
            f"Region et={et}, eta={eta} already trained. Skipping dataset and split manifest loading."
        )
        return True

    def _get_split_manifest(self) -> SplitManifest:
        """Lazily instantiates and caches the SplitManifest for this run."""
        manifest_path = self.config.split_manifest_path or os.path.join(
            self.results_folder_path, "split_manifest.json"
        )
        if (
            self._split_manifest is None
            or str(self._split_manifest.manifest_file) != str(manifest_path)
        ):
            self._split_manifest = SplitManifest(manifest_path)
        return self._split_manifest

    @property
    def device(self) -> torch.device:
        return self._device

    @device.setter
    def device(self, value: torch.device) -> None:
        self._device = value

    @property
    def full_dataset(self) -> Any:
        return self._full_dataset

    @full_dataset.setter
    def full_dataset(self, value: Any) -> None:
        self._full_dataset = value
        self._apply_debug_overrides()

    def run(self) -> None:
        """Runs training across all specified transverse energy (ET) and pseudorapidity (eta) regions."""
        if self.config.results_folder_path is None:
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            folder_name = f"config_name{self.config.config_name}_id{timestamp}"
            self.results_folder_path = str(create_folder(folder_name))
        else:
            self.results_folder_path = str(self.config.results_folder_path)
            os.makedirs(self.results_folder_path, exist_ok=True)

        eta_et_region = list(
            ParameterGrid(
                {
                    "eta": list(self.config.eta_range_idx),
                    "et": list(self.config.et_range_idx),
                }
            )
        )
        for index in range(len(self.full_dataset)):
            self.et, self.eta = get_et_eta(self.full_dataset.file_paths[index])
            if {
                "eta": int(self.eta),
                "et": int(self.et),
            } in eta_et_region:
                self.train_region(index)
            if self.config.debug:
                break

    def train_region(self, index: int) -> str:
        """Executes full training repeats and cross-validation folds for a single kinematic region.

        Args:
            index: Dataset file index to train.

        Returns:
            Source file path of the processed region dataset.
        """
        et = int(self.et)
        eta = int(self.eta)
        if self._is_region_complete(et, eta):
            return str(self.full_dataset.file_paths[index])

        data, target, path = self.full_dataset[index]
        manifest = self._get_split_manifest()
        region_key = f"et_{et}_eta_{eta}"

        test_data, train_cross_validation = manifest.get_or_create_region_splits(
            region_key, data, target, self.kfold
        )

        total_folds = len(train_cross_validation)
        ring_indices = self.full_dataset.ring_column_indices

        for fold_idx, (train_index, val_index) in enumerate(train_cross_validation):
            missing_repeats = self._get_missing_repeats_for_fold(et, eta, fold_idx)
            if not missing_repeats:
                continue

            train_dl = self.factory.create_dataloader(
                data[train_index][:, ring_indices], target[train_index]
            )
            val_dl = self.factory.create_dataloader(
                data[val_index][:, ring_indices], target[val_index]
            )

            for repeat in missing_repeats:
                log.info(
                    f"Executing fold: {fold_idx + 1}/{total_folds}. Repeat: {repeat + 1}"
                )
                self.recorder.clear()

                model = self.factory.create_model()
                optimizer = self.factory.create_optimizer(model)
                loss_fn = self.factory.create_loss_function()

                trainer = FoldTrainer(
                    model, optimizer, loss_fn, self.device, self.config.pred_target_limiar
                )
                callback = SPCallbackPyTorch(patience=25, verbose=True)

                results = trainer.fit(
                    train_dl, val_dl, self.config.epochs, callback
                )

                if results["best_weights"] is not None:
                    hyperparameters = {
                        "optimizer": self.config.optimizer_function.object_name,
                        "optimizer_params": {
                            k: v
                            for k, v in self.config.optimizer_function.parameters.items()
                            if k != "params"
                        },
                        "loss": self.config.loss_function.object_name,
                        "loss_params": self.config.loss_function.parameters,
                        "batch_size": self.config.batch_size,
                    }
                    self.recorder.record(
                        repeat + 1,
                        path,
                        fold_idx,
                        results["best_sp_value"],
                        results["best_fa_value"],
                        results["best_pd_value"],
                        results["best_weights"],
                        results["history"],
                        parameters=hyperparameters,
                    )

                self.recorder.save(
                    self.results_folder_path, self.et, self.eta, repeat, fold_idx
                )

        return path
