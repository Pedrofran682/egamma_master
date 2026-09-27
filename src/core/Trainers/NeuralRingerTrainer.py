import logging
import os
from datetime import datetime
from typing import Any, List, Tuple
import numpy as np
import torch
from sklearn.model_selection import ParameterGrid

from src.core.Callbacks.SPCallbackPyTorch import SPCallbackPyTorch
from src.core.Datasets.EgammaNpzDataset import EgammaNpzDataset
from src.core.Datasets.SplitManifest import SplitManifest
from src.core.Trainers.FoldTrainer import FoldTrainer
from src.core.Trainers.ResultsRecorder import ResultsRecorder
from src.core.Trainers.TrainingFactory import TrainingFactory
from src.Parser.NeuralRingerTrainerConfiguration import NeuralRingerTrainerConfiguration
from src.utils import create_folder, get_et_eta, get_instance, verify_results

log = logging.getLogger()


class NeuralRingerTrainer:
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
        self.use_cuda: bool = torch.cuda.is_available()
        self.device: torch.device = torch.device("cuda" if self.use_cuda else "cpu")
        self.factory: TrainingFactory = TrainingFactory(self.config, self.device)
        self.recorder: ResultsRecorder = ResultsRecorder()
        self.kfold: Any = self.factory.create_cross_validation()
        self.et: int = -1
        self.eta: int = -1
        self.full_dataset: EgammaNpzDataset = get_instance(self.config.dataset)
        self.full_dataset.config = config
        self.results_folder_path: str = ""

        if self.config.debug:
            log.warning("#### EXECUTING ON DEBUG MODE. ####")
            self.config.epochs = 2
            self.config.n_initializations = 1

    def run(self) -> None:
        """Runs training across all specified transverse energy (ET) and pseudorapidity (eta) regions."""
        if self.config.results_folder_path is None:
            folder_name = "config_name{config_name}_id{id}".format(
                config_name=self.config.config_name,
                id=datetime.now().strftime("%Y%m%d%H%M%S"),
            )
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
        data, target, path = self.full_dataset[index]
        manifest_path = os.path.join(self.results_folder_path, "split_manifest.json")
        manifest = SplitManifest(manifest_path)
        region_key = f"et_{int(self.et)}_eta_{int(self.eta)}"

        test_data, train_cross_validation = manifest.get_or_create_region_splits(
            region_key, data, target, self.kfold
        )

        total_folds = len(train_cross_validation)
        ring_indices = self.full_dataset.ring_column_indices

        for fold_idx, (train_index, val_index) in enumerate(train_cross_validation):
            train_dl = self.factory.create_dataloader(
                data[train_index][:, ring_indices], target[train_index]
            )
            val_dl = self.factory.create_dataloader(
                data[val_index][:, ring_indices], target[val_index]
            )

            for repeat in range(self.config.n_initializations):
                if verify_results(
                    self.results_folder_path, self.et, self.eta, repeat, fold_idx
                ):
                    continue

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
                callback = SPCallbackPyTorch(patience=10, verbose=True)

                results = trainer.fit(
                    train_dl, val_dl, self.config.epochs, callback
                )

                if results["best_weights"] is not None:
                    self.recorder.record(
                        repeat + 1,
                        path,
                        fold_idx,
                        results["best_sp_value"],
                        results["best_fa_value"],
                        results["best_pd_value"],
                        results["best_weights"],
                        results["history"],
                    )

                self.recorder.save(
                    self.results_folder_path, self.et, self.eta, repeat, fold_idx
                )

        return path


    def generate_folds_with_holdout(
        self, features: np.ndarray, labels: np.ndarray, test_fold_idx: int = 0
    ) -> Tuple[np.ndarray, List[Tuple[np.ndarray, np.ndarray]]]:
        """Splits features and labels into a holdout test partition and cross-validation iterable.

        Args:
            features: Input features array.
            labels: Binary labels array.
            test_fold_idx: Index of the fold selected as holdout test.

        Returns:
            Tuple of (holdout_indices, cv_train_val_iterable).

        Raises:
            ValueError: If test_fold_idx is outside split boundaries.
        """
        all_folds = [test_idx for _, test_idx in self.kfold.split(features, labels)]
        if not (0 <= test_fold_idx < self.kfold.get_n_splits()):
            raise ValueError(
                f"test_fold_idx must be between 0 and {self.kfold.get_n_splits() - 1}"
            )

        holdout_indices = all_folds[test_fold_idx]
        remaining_folds = [
            fold for index, fold in enumerate(all_folds) if index != test_fold_idx
        ]
        cv_iterable = []
        for index in range(len(remaining_folds)):
            val_indices = remaining_folds[index]
            train_folds = [fold for j, fold in enumerate(remaining_folds) if j != index]
            train_indices = np.concatenate(train_folds)
            cv_iterable.append((train_indices, val_indices))

        return holdout_indices, cv_iterable
