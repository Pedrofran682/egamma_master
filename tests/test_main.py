import tempfile
import unittest
from unittest.mock import MagicMock, patch
import yaml

from main import build_parser, main


class TestMain(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_build_parser_defaults(self):
        parser = build_parser()
        args = parser.parse_args(["--config", "config/test.yaml"])
        self.assertEqual(args.config, "config/test.yaml")
        self.assertIsNone(args.results_path)

    def test_build_parser_with_results_path(self):
        parser = build_parser()
        args = parser.parse_args(["--config", "config/test.yaml", "--results_path", "results/custom_run"])
        self.assertEqual(args.config, "config/test.yaml")
        self.assertEqual(args.results_path, "results/custom_run")

    @patch("main.NeuralRingerTrainer")
    def test_main_overrides_results_folder_path(self, mock_trainer_cls):
        yaml_content = {
            "config_name": "test_exp",
            "batch_size": 32,
            "epochs": 5,
            "et_range_idx": [0],
            "eta_range_idx": [0],
            "n_initializations": 1,
            "pred_target_limiar": 0.5,
            "results_folder_path": "results/original_path",
            "loss_function": {"module": "torch.nn", "object_name": "BCELoss", "parameters": {}},
            "optimizer_function": {"module": "torch.optim", "object_name": "Adam", "parameters": {"lr": 0.001}},
            "kFold": {"module": "sklearn.model_selection", "object_name": "StratifiedKFold", "parameters": {"n_splits": 5}},
            "model": {"module": "tests.test_trainers", "object_name": "SimpleTestModel", "parameters": {"input_dim": 10}},
            "dataset": {"module": "src.core.Datasets.EgammaNpzDataset", "object_name": "EgammaNpzDataset", "parameters": {"drive_path": "data/"}},
        }
        yaml_file = f"{self.temp_dir.name}/config.yaml"
        with open(yaml_file, "w") as f:
            yaml.dump(yaml_content, f)

        mock_trainer_instance = MagicMock()
        mock_trainer_cls.return_value = mock_trainer_instance

        parser = build_parser()
        args = parser.parse_args(["--config", yaml_file, "--results_path", "results/overridden_path"])

        main(args)

        mock_trainer_cls.assert_called_once()
        passed_config = mock_trainer_cls.call_args[0][0]
        self.assertEqual(passed_config.results_folder_path, "results/overridden_path")
        mock_trainer_instance.run.assert_called_once()


if __name__ == "__main__":
    unittest.main()
