import os
import tempfile
import unittest
from typing import Dict

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


if __name__ == "__main__":
    unittest.main()
