import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch
import pandas as pd
import yaml

from src.core.Interfaces.BaseHyperparameterTuner import BaseHyperparameterTuner
from src.core.Tuning.GridSearchTuner import GridSearchTuner
from src.core.Tuning.TuningConfiguration import (
    LossSearchItem,
    OptimizerSearchItem,
    TuningConfiguration,
    TuningSearchSpace,
)


class TestGridSearchTuner(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.base_config_path = os.path.join(self.temp_dir.name, "base_config.yaml")

        base_config_data = {
            "config_name": "TestModel",
            "batch_size": 64,
            "epochs": 10,
            "n_initializations": 1,
            "pred_target_limiar": 0.5,
            "et_range_idx": [0, 1],
            "eta_range_idx": [0, 1],
            "loss_function": {
                "module": "torch.nn",
                "object_name": "BCELoss",
                "parameters": {},
            },
            "optimizer_function": {
                "module": "torch.optim",
                "object_name": "Adam",
                "parameters": {"lr": 0.01},
            },
            "kFold": {
                "module": "sklearn.model_selection",
                "object_name": "StratifiedKFold",
                "parameters": {"n_splits": 2},
            },
            "model": {
                "module": "src.Models.egamma.ModelV6",
                "object_name": "ModelV6",
                "parameters": {"input_dim": 50},
            },
            "dataset": {
                "module": "src.core.Datasets.EgammaNpzDataset",
                "object_name": "EgammaNpzDataset",
                "parameters": {
                    "drive_path": self.temp_dir.name,
                    "startswith": "data",
                    "endswith": ".npz",
                    "percentage": 1.0,
                },
            },
        }

        with open(self.base_config_path, "w") as f:
            yaml.safe_dump(base_config_data, f)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_tuning_configuration_generation(self) -> None:
        tuning_config = TuningConfiguration(
            base_config=self.base_config_path,
            overrides={"et_range_idx": [1], "eta_range_idx": [1]},
            search_space=TuningSearchSpace(
                loss_functions=[LossSearchItem(object_name="BCELoss")],
                optimizers=[
                    OptimizerSearchItem(
                        object_name="Adam",
                        parameters={"lr": [0.001, 0.01], "weight_decay": [0.0, 0.001]},
                    ),
                    OptimizerSearchItem(
                        object_name="SGD",
                        parameters={"lr": [0.05], "momentum": [0.9]},
                    ),
                ],
            ),
            output_dir=os.path.join(self.temp_dir.name, "tuning_results"),
        )

        trials = tuning_config.build_trial_configurations()
        self.assertEqual(len(trials), 5)

        for tag, cfg, meta in trials:
            self.assertEqual(cfg.et_range_idx, [1])
            self.assertEqual(cfg.eta_range_idx, [1])
            self.assertEqual(cfg.loss_function.object_name, "BCELoss")
            self.assertIn("trial_", tag)

    def test_grid_search_tuner_interface_and_run(self) -> None:
        tuning_config = TuningConfiguration(
            base_config=self.base_config_path,
            overrides={"et_range_idx": [1], "eta_range_idx": [1]},
            search_space=TuningSearchSpace(
                optimizers=[
                    OptimizerSearchItem(
                        object_name="Adam",
                        parameters={"lr": [0.001, 0.01]},
                    )
                ]
            ),
            output_dir=os.path.join(self.temp_dir.name, "tuning_results"),
        )

        tuner = GridSearchTuner(tuning_config)
        self.assertIsInstance(tuner, BaseHyperparameterTuner)

        trials = tuner.generate_trials()
        self.assertEqual(len(trials), 2)

        with patch("src.core.Tuning.GridSearchTuner.NeuralRingerTrainer") as mock_trainer_cls, \
             patch("src.core.Tuning.GridSearchTuner.ResultAggregator") as mock_aggregator_cls:

            mock_trainer = MagicMock()
            mock_trainer_cls.return_value = mock_trainer

            mock_aggregator = MagicMock()
            mock_aggregator.has_region.return_value = True
            mock_aggregator.get_best_model_for_region.side_effect = [
                ({"best_sp_value": 0.95, "best_pd_value": 0.94, "best_fa_value": 0.05}, 1, 0.93, 0.01),
                ({"best_sp_value": 0.98, "best_pd_value": 0.97, "best_fa_value": 0.03}, 1, 0.96, 0.005),
            ]
            mock_aggregator_cls.return_value = mock_aggregator

            results_df = tuner.run()

            self.assertEqual(len(results_df), 2)
            self.assertIn("mean_sp", results_df.columns)
            self.assertGreaterEqual(results_df.loc[0, "mean_sp"], results_df.loc[1, "mean_sp"])

    def test_tuning_with_split_manifest_path(self) -> None:
        manifest_file = os.path.join(self.temp_dir.name, "custom_manifest.json")
        tuning_config = TuningConfiguration(
            base_config=self.base_config_path,
            split_manifest_path=manifest_file,
            search_space=TuningSearchSpace(
                optimizers=[OptimizerSearchItem(object_name="Adam", parameters={"lr": [0.001]})]
            ),
        )
        trials = tuning_config.build_trial_configurations()
        self.assertEqual(len(trials), 1)
        self.assertEqual(trials[0][1].split_manifest_path, manifest_file)

    def test_load_real_tuning_yaml_config(self) -> None:
        real_yaml = "config/tuning/ModelV6_50Rings_region1_1_tuning.yaml"
        self.assertTrue(os.path.exists(real_yaml))

        config = TuningConfiguration.from_yaml(real_yaml)
        trials = config.build_trial_configurations()

        self.assertGreater(len(trials), 0)
        for tag, cfg, meta in trials:
            self.assertEqual(cfg.et_range_idx, [1])
            self.assertEqual(cfg.eta_range_idx, [1])
            self.assertEqual(cfg.loss_function.object_name, "BCELoss")
            self.assertIn("ModelV6", cfg.model.object_name)

