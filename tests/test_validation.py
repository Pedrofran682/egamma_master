import os
import pathlib
import tempfile
import unittest
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src.core.Validation.EfficiencyPlotter import EfficiencyPlotter
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
        import shutil
        shutil.rmtree("Plots/dummy", ignore_errors=True)
        shutil.rmtree("Plots/ModelV1_Run123", ignore_errors=True)

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

        self.assertTrue(aggregator.has_region(1, 1))
        self.assertFalse(aggregator.has_region(2, 2))
        best_model_reg, best_rep_reg, mean_sp_reg, std_sp_reg = aggregator.get_best_model_for_region(1, 1)
        self.assertEqual(best_rep_reg, 2)
        self.assertEqual(best_model_reg["best_sp_value"], 0.94)
        with self.assertRaises(KeyError):
            aggregator.get_best_model_for_region(2, 2)

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
                # Default plot_dir: Plots/<results_folder_name>
                validator_default = ModelValidator(
                    config_path="config/NeuralRinger/ModelV1.yaml",
                    data_path="results/ModelV1_Run123",
                )
                self.assertEqual(
                    validator_default.plot_dir,
                    pathlib.Path("Plots/ModelV1_Run123"),
                )

                # Custom plot_dir via output_dir argument
                custom_dir = os.path.join(self.temp_dir.name, "CustomValidationPlots")
                validator_custom = ModelValidator(
                    config_path="config/NeuralRinger/ModelV1_Run123.yaml",
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

    def test_plot_regional_efficiency(self):
        evaluator = HoldoutEvaluator(default_target_pd=0.90)
        result = evaluator.evaluate(
            model=self.model,
            data=self.features,
            target=self.labels,
            test_indices=np.arange(40),
            ring_column_indices=np.arange(10),
            device=torch.device("cpu"),
            target_pd=0.90,
        )
        plotter = EfficiencyPlotter(et_index=1, eta_index=2)
        out_dir = pathlib.Path(self.temp_dir.name) / "RegionPlots"
        out_dir.mkdir(parents=True, exist_ok=True)
        plotter.plot_regional_efficiency(
            result, iet=1, ieta=2, plot_dir=out_dir, config_name="TestModel"
        )

        self.assertTrue(
            (out_dir / "TestModel_iet1_ieta2_signal_efficiency_vs_et_standard_cut.png").exists()
        )
        self.assertTrue(
            (out_dir / "TestModel_iet1_ieta2_signal_efficiency_vs_et_target_pd_cut.png").exists()
        )
        self.assertTrue(
            (out_dir / "TestModel_iet1_ieta2_signal_efficiency_vs_eta_standard_cut.png").exists()
        )
        self.assertTrue(
            (out_dir / "TestModel_iet1_ieta2_signal_efficiency_vs_eta_target_pd_cut.png").exists()
        )

    def test_model_validator_run_routes_validation_plots_to_subfolder(self):
        from unittest.mock import MagicMock, patch
        from src.core.Validation.ModelValidator import ModelValidator

        mock_config = MagicMock()
        mock_config.config_name = "TestConfig"
        with patch.object(ModelValidator, "_load_config", return_value=mock_config):
            with patch("src.core.Validation.ModelValidator.NeuralRingerTrainer"):
                custom_dir = os.path.join(self.temp_dir.name, "ValidationRunTest")
                run_dir = os.path.join(self.temp_dir.name, "RunDir")
                os.makedirs(run_dir, exist_ok=True)
                manifest_path = os.path.join(run_dir, "split_manifest.json")
                with open(manifest_path, "w") as f:
                    f.write("{}")

                validator = ModelValidator(
                    config_path="config/TestModel.yaml",
                    data_path=run_dir,
                    output_dir=custom_dir,
                )
                validator.trainer = MagicMock()
                validator.trainer.full_dataset = []
                validator.efficiency_plotter = MagicMock()

                validator.run()

                expected_validation_dir = pathlib.Path(custom_dir) / "Validation"
                self.assertTrue(expected_validation_dir.exists())

    def test_model_validator_raises_when_manifest_missing(self):
        from unittest.mock import MagicMock, patch
        from src.core.Validation.ModelValidator import ModelValidator

        mock_config = MagicMock()
        with patch.object(ModelValidator, "_load_config", return_value=mock_config):
            with patch("src.core.Validation.ModelValidator.NeuralRingerTrainer"):
                empty_run_dir = os.path.join(self.temp_dir.name, "EmptyRunDir")
                os.makedirs(empty_run_dir, exist_ok=True)
                validator = ModelValidator(
                    config_path="config/TestModel.yaml",
                    data_path=empty_run_dir,
                )
                with self.assertRaises(FileNotFoundError):
                    validator.run()

    def test_model_validator_retrieves_exact_manifest_test_indices(self):
        import json
        from unittest.mock import MagicMock, patch
        from src.core.Validation.ModelValidator import ModelValidator

        mock_config = MagicMock()
        mock_config.config_name = "TestConfig"
        with patch.object(ModelValidator, "_load_config", return_value=mock_config):
            with patch("src.core.Validation.ModelValidator.NeuralRingerTrainer") as mock_trainer_cls:
                mock_trainer = MagicMock()
                mock_trainer_cls.return_value = mock_trainer
                mock_trainer.full_dataset.file_paths = ["consolidated.et1.eta1.npz"]
                mock_trainer.full_dataset.ring_column_indices = np.arange(10)
                mock_trainer.full_dataset.__len__.return_value = 1
                mock_trainer.full_dataset.__getitem__.return_value = (
                    np.zeros((50, 10), dtype=np.float32),
                    np.zeros(50, dtype=np.int32),
                    "consolidated.et1.eta1.npz",
                )

                run_dir = os.path.join(self.temp_dir.name, "ManifestRunDir")
                os.makedirs(run_dir, exist_ok=True)
                manifest_path = os.path.join(run_dir, "split_manifest.json")
                expected_test_indices = [2, 5, 8, 12]
                with open(manifest_path, "w") as f:
                    json.dump(
                        {
                            "et_1_eta_1": {
                                "test_indices": expected_test_indices,
                                "cv_splits": [],
                            }
                        },
                        f,
                    )

                validator = ModelValidator(
                    config_path="config/TestModel.yaml",
                    data_path=run_dir,
                    output_dir=os.path.join(self.temp_dir.name, "OutputPlots"),
                )
                validator.aggregator = MagicMock()
                validator.aggregator.has_region.return_value = True
                validator.aggregator.get_best_model_for_region.return_value = (
                    {"best_weights": {}, "history": {"callbackMetrics": {}}},
                    0,
                    0.95,
                    0.0,
                )
                validator.evaluator = MagicMock()
                validator.roc_plotter = MagicMock()
                validator.profile_plotter = MagicMock()
                validator.efficiency_plotter = MagicMock()

                validator.run()

                validator.evaluator.evaluate.assert_called_once()
                call_kwargs = validator.evaluator.evaluate.call_args[1]
                np.testing.assert_array_equal(call_kwargs["test_indices"], np.array(expected_test_indices))
                validator.efficiency_plotter.plot_regional_efficiency.assert_called_once()
                validator.efficiency_plotter.generate_global_plots.assert_called_once()

    def test_model_validator_raises_when_region_missing_in_manifest(self):
        import json
        from unittest.mock import MagicMock, patch
        from src.core.Validation.ModelValidator import ModelValidator

        mock_config = MagicMock()
        with patch.object(ModelValidator, "_load_config", return_value=mock_config):
            with patch("src.core.Validation.ModelValidator.NeuralRingerTrainer") as mock_trainer_cls:
                mock_trainer = MagicMock()
                mock_trainer_cls.return_value = mock_trainer
                mock_trainer.full_dataset.file_paths = ["consolidated.et1.eta1.npz"]
                mock_trainer.full_dataset.__len__.return_value = 1

                run_dir = os.path.join(self.temp_dir.name, "MissingRegionRunDir")
                os.makedirs(run_dir, exist_ok=True)
                manifest_path = os.path.join(run_dir, "split_manifest.json")
                with open(manifest_path, "w") as f:
                    json.dump({"et_2_eta_2": {"test_indices": [1], "cv_splits": []}}, f)

                validator = ModelValidator(
                    config_path="config/TestModel.yaml",
                    data_path=run_dir,
                )
                validator.aggregator = MagicMock()
                validator.aggregator.has_region.return_value = True

                with self.assertRaises(KeyError):
                    validator.run()


if __name__ == "__main__":
    unittest.main()
