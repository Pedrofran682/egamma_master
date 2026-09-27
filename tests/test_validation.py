import os
import pathlib
import tempfile
import unittest
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src.core.Validation.HoldoutEvaluator import HoldoutEvaluator
from src.core.Validation.ResultAggregator import ResultAggregator


class DummyClassifier(nn.Module):
    def __init__(self, input_dim: int = 10):
        super().__init__()
        self.fc = nn.Linear(input_dim, 1)
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.fc(x))


class TestValidation(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.features = np.random.randn(100, 10).astype(np.float32)
        self.labels = np.random.choice([0, 1], size=100).astype(np.int32)
        self.temp_dir = tempfile.TemporaryDirectory()
        self.model = DummyClassifier(input_dim=10)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_group_and_concat(self):
        record_1 = [
            {"reapet": 1, "fold": 0, "best_sp_value": 0.88, "best_weights": {}},
            {"reapet": 1, "fold": 1, "best_sp_value": 0.90, "best_weights": {}},
        ]
        record_2 = [
            {"reapet": 2, "fold": 0, "best_sp_value": 0.92, "best_weights": {}},
            {"reapet": 2, "fold": 1, "best_sp_value": 0.94, "best_weights": {}},
        ]

        file1 = os.path.join(self.temp_dir.name, "repeat1.fold_idx0.iet1.ieta1.pkl")
        file2 = os.path.join(self.temp_dir.name, "repeat2.fold_idx0.iet1.ieta1.pkl")

        pd.to_pickle(record_1, file1)
        pd.to_pickle(record_2, file2)

        aggregator = ResultAggregator(self.temp_dir.name)
        grouped = aggregator.group_and_concat()

        self.assertIn("iet1.ieta1", grouped)
        df = grouped["iet1.ieta1"]
        self.assertEqual(len(df), 4)

        best_model, best_rep, mean_sp, std_sp = aggregator.get_best_model_details(df)
        self.assertEqual(best_rep, 2)
        self.assertAlmostEqual(mean_sp, 0.93, places=2)
        self.assertEqual(best_model["best_sp_value"], 0.94)

    def test_evaluate_holdout(self):
        test_indices = np.arange(40)
        ring_indices = np.arange(10)

        evaluator = HoldoutEvaluator(default_target_pd=0.90)
        result = evaluator.evaluate(
            model=self.model,
            data=self.features,
            target=self.labels,
            test_indices=test_indices,
            ring_column_indices=ring_indices,
            device=torch.device("cpu"),
            target_pd=0.90,
        )

        self.assertTrue(0.0 <= result.global_acc <= 1.0)
        self.assertTrue(0.0 <= result.signal_acc <= 1.0)
        self.assertTrue(0.0 <= result.bg_acc <= 1.0)
        self.assertTrue(0.0 <= result.pf <= 1.0)
        self.assertEqual(len(result.preds_05), 40)
        self.assertEqual(len(result.preds_cut), 40)

    def test_model_validator_default_and_custom_plot_dir(self):
        from unittest.mock import MagicMock, patch
        from src.core.Validation.ModelValidator import ModelValidator
        import scripts.validate_pd as validate_pd_module

        with patch.object(ModelValidator, "_load_config", return_value=MagicMock()):
            with patch("src.core.Validation.ModelValidator.NeuralRingerTrainer"):
                # Default plot_dir: Plots/<results_name>/Validation
                validator_default = ModelValidator(
                    config_path="dummy.yaml",
                    data_path="results/ModelV1_Run123",
                )
                self.assertEqual(
                    validator_default.plot_dir,
                    pathlib.Path("Plots/ModelV1_Run123/Validation"),
                )

                # Custom plot_dir via output_dir argument
                custom_dir = os.path.join(self.temp_dir.name, "CustomValidationPlots")
                validator_custom = ModelValidator(
                    config_path="dummy.yaml",
                    data_path="results/ModelV1_Run123",
                    output_dir=custom_dir,
                )
                self.assertEqual(validator_custom.plot_dir, pathlib.Path(custom_dir))
                self.assertTrue(validator_custom.plot_dir.exists())

        # Test CLI parser in validate_pd.py
        parsed = validate_pd_module.parser.parse_args(
            ["--config", "conf.yaml", "--data_path", "results/test", "--output_dir", "Plots/custom"]
        )
        self.assertEqual(parsed.output_dir, "Plots/custom")


if __name__ == "__main__":
    unittest.main()
