import os
import tempfile
import unittest
from typing import Dict
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.core.Callbacks.SPCallbackPyTorch import SPCallbackPyTorch
from src.core.Trainers.FoldTrainer import FoldTrainer
from src.core.Trainers.NeuralRingerTrainer import NeuralRingerTrainer
from src.core.Trainers.ResultsRecorder import ResultsRecorder
from src.core.Trainers.TrainingFactory import TrainingFactory
from src.core.Validation.ResultAggregator import ResultAggregator
from src.Parser.DynamicConfiguration import DynamicConfiguration
from src.Parser.NeuralRingerTrainerConfiguration import NeuralRingerTrainerConfiguration
from src.utils import get_results_file_name


class TinyClassifier(nn.Module):
    """Minimal binary classifier used for trainer component unit tests."""

    def __init__(self, input_dim: int = 4) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)
        self.activation = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.linear(x))


class TestTrainerComponents(unittest.TestCase):
    """Unit tests for trainer cleanup, immutability, and compatibility fixes."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _build_config(
        self,
        *,
        debug: bool = False,
        epochs: int = 20,
        n_initializations: int = 4,
        optimizer_params: Dict[str, float] | None = None,
    ) -> NeuralRingerTrainerConfiguration:
        return NeuralRingerTrainerConfiguration(
            config_name="test_trainer_components",
            batch_size=8,
            num_workers=0,
            balance_data=False,
            debug=debug,
            epochs=epochs,
            n_initializations=n_initializations,
            et_range_idx=[0],
            eta_range_idx=[0],
            pred_target_limiar=0.5,
            loss_function=DynamicConfiguration(
                module="torch.nn",
                object_name="BCELoss",
                parameters={},
            ),
            optimizer_function=DynamicConfiguration(
                module="torch.optim",
                object_name="Adam",
                parameters=optimizer_params if optimizer_params is not None else {"lr": 0.005},
            ),
            kFold=DynamicConfiguration(
                module="sklearn.model_selection",
                object_name="StratifiedKFold",
                parameters={"n_splits": 3},
            ),
            model=DynamicConfiguration(
                module="tests.core.trainers.test_trainer_components",
                object_name="TinyClassifier",
                parameters={"input_dim": 4},
            ),
            dataset=DynamicConfiguration(
                module="src.core.Datasets.EgammaNpzDataset",
                object_name="EgammaNpzDataset",
                parameters={"drive_path": self.temp_dir.name},
            ),
        )

    def test_neural_ringer_trainer_debug_overrides_at_init(self) -> None:
        config = self._build_config(debug=True, epochs=50, n_initializations=5)
        trainer = NeuralRingerTrainer(config)

        self.assertEqual(trainer.config.epochs, 2)
        self.assertEqual(trainer.config.n_initializations, 1)

    def test_neural_ringer_trainer_debug_overrides_on_dataset_setter(self) -> None:
        config = self._build_config(debug=True, epochs=50, n_initializations=5)
        trainer = NeuralRingerTrainer(config)
        trainer.config.epochs = 15
        trainer.config.n_initializations = 3

        trainer.full_dataset = trainer.full_dataset

        self.assertEqual(trainer.config.epochs, 2)
        self.assertEqual(trainer.config.n_initializations, 1)

    def test_neural_ringer_trainer_non_debug_and_removed_dead_method(self) -> None:
        config = self._build_config(debug=False, epochs=15, n_initializations=3)
        trainer = NeuralRingerTrainer(config)

        self.assertEqual(trainer.config.epochs, 15)
        self.assertEqual(trainer.config.n_initializations, 3)
        self.assertFalse(hasattr(trainer, "generate_folds_with_holdout"))

    def test_training_factory_create_optimizer_does_not_mutate_config(self) -> None:
        config = self._build_config(optimizer_params={"lr": 0.005, "weight_decay": 0.01})
        factory = TrainingFactory(config, device=torch.device("cpu"))

        first_model = TinyClassifier(input_dim=4)
        second_model = TinyClassifier(input_dim=4)

        first_optimizer = factory.create_optimizer(first_model)
        second_optimizer = factory.create_optimizer(second_model)

        self.assertIsInstance(first_optimizer, torch.optim.Adam)
        self.assertIsInstance(second_optimizer, torch.optim.Adam)
        self.assertAlmostEqual(first_optimizer.defaults["lr"], 0.005)
        self.assertAlmostEqual(first_optimizer.defaults["weight_decay"], 0.01)
        self.assertNotIn("params", config.optimizer_function.parameters)
        self.assertEqual(
            config.optimizer_function.parameters,
            {"lr": 0.005, "weight_decay": 0.01},
        )

    def test_sp_callback_omits_unused_conv_attributes(self) -> None:
        callback = SPCallbackPyTorch(patience=3, verbose=False)

        self.assertFalse(hasattr(callback, "kernel_size"))
        self.assertFalse(hasattr(callback, "num_filters"))

    def test_fold_trainer_history_omits_reapet_key(self) -> None:
        torch.manual_seed(0)
        features = torch.randn(20, 4)
        targets = torch.tensor([0.0, 1.0] * 10).view(-1, 1)
        loader = DataLoader(TensorDataset(features, targets), batch_size=5)

        model = TinyClassifier(input_dim=4)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        trainer = FoldTrainer(model, optimizer, nn.BCELoss(), torch.device("cpu"))
        callback = SPCallbackPyTorch(patience=2, verbose=False)

        results = trainer.fit(loader, loader, epochs=1, callback=callback)

        self.assertNotIn("reapet", results["history"])

    def test_results_recorder_and_aggregator_support_repeat_and_legacy_reapet(self) -> None:
        recorder = ResultsRecorder()
        recorder.record(
            repeat=1,
            file_path="mock_iet1_ieta1.npz",
            fold=0,
            best_sp=0.85,
            best_fa=0.10,
            best_pd=0.90,
            best_weights={},
            history={},
        )
        saved_new = recorder.save(self.temp_dir.name, et=1, eta=1, repeat=1, fold_idx=0)
        saved_df = pd.read_pickle(saved_new)
        self.assertEqual(saved_df.iloc[0]["repeat"], 1)

        legacy_file = os.path.join(self.temp_dir.name, "repeat2.fold_idx0.iet1.ieta1.pkl")
        pd.to_pickle(
            [{"reapet": 2, "fold": 0, "best_sp_value": 0.93, "best_weights": {}}],
            legacy_file,
        )

        aggregator = ResultAggregator(self.temp_dir.name)
        best_model, best_rep, mean_sp, _ = aggregator.get_best_model_for_region(1, 1)
        self.assertEqual(best_rep, 2)
        self.assertAlmostEqual(mean_sp, 0.93)
        self.assertEqual(best_model["repeat"], 2)

        repeat_only_df = pd.DataFrame(
            [
                {"repeat": 1, "fold": 0, "best_sp_value": 0.80, "best_weights": {}},
                {"repeat": 2, "fold": 0, "best_sp_value": 0.91, "best_weights": {}},
            ]
        )
        _, best_rep_new, mean_sp_new, _ = ResultAggregator.get_best_model_details(repeat_only_df)
        self.assertEqual(best_rep_new, 2)
        self.assertAlmostEqual(mean_sp_new, 0.91)

    def test_train_region_skips_dataset_and_manifest_when_region_complete(self) -> None:
        config = self._build_config(epochs=1, n_initializations=1)
        trainer = NeuralRingerTrainer(config)
        trainer.results_folder_path = self.temp_dir.name
        trainer.et = 0
        trainer.eta = 0

        expected_path = os.path.join(self.temp_dir.name, "sample.et0.eta0.npz")
        mock_dataset = MagicMock()
        mock_dataset.file_paths = [expected_path]
        trainer.full_dataset = mock_dataset

        for fold_idx in range(2):
            result_file = os.path.join(
                self.temp_dir.name, get_results_file_name(0, 0, 0, fold_idx)
            )
            pd.to_pickle([{"repeat": 1, "fold": fold_idx, "best_sp_value": 0.9}], result_file)

        with patch("src.core.Trainers.NeuralRingerTrainer.SplitManifest") as mock_manifest_cls:
            returned_path = trainer.train_region(0)

        self.assertEqual(returned_path, expected_path)
        mock_dataset.__getitem__.assert_not_called()
        mock_manifest_cls.assert_not_called()

    def test_train_region_skips_completed_fold_dataloader_and_caches_manifest(self) -> None:
        torch.manual_seed(0)
        np.random.seed(0)
        config = self._build_config(epochs=1, n_initializations=1)
        config.split_manifest_path = os.path.join(self.temp_dir.name, "split_manifest.json")
        trainer = NeuralRingerTrainer(config)
        trainer.results_folder_path = self.temp_dir.name
        trainer.et = 0
        trainer.eta = 0

        existing_fold0 = os.path.join(
            self.temp_dir.name, get_results_file_name(0, 0, 0, 0)
        )
        pd.to_pickle([{"repeat": 1, "fold": 0, "best_sp_value": 0.9}], existing_fold0)

        features = np.random.randn(30, 4).astype(np.float32)
        labels = np.array([0, 1] * 15, dtype=np.int32)
        dataset_path = os.path.join(self.temp_dir.name, "sample.et0.eta0.npz")
        mock_dataset = MagicMock()
        mock_dataset.file_paths = [dataset_path]
        mock_dataset.ring_column_indices = [0, 1, 2, 3]
        mock_dataset.__getitem__.return_value = (features, labels, dataset_path)
        trainer.full_dataset = mock_dataset

        with patch.object(
            trainer.factory,
            "create_dataloader",
            wraps=trainer.factory.create_dataloader,
        ) as mock_create_dl:
            returned_path = trainer.train_region(0)

        self.assertEqual(returned_path, dataset_path)
        self.assertEqual(mock_create_dl.call_count, 2)
        self.assertTrue(
            os.path.exists(
                os.path.join(self.temp_dir.name, get_results_file_name(0, 0, 0, 1))
            )
        )
        first_manifest = trainer._get_split_manifest()
        second_manifest = trainer._get_split_manifest()
        self.assertIs(first_manifest, second_manifest)
        self.assertIn("et_0_eta_0", first_manifest.data)


if __name__ == "__main__":
    unittest.main()
