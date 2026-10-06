import os
import pathlib
import tempfile
import unittest
from unittest.mock import MagicMock, patch
import numpy as np

from src.core.Plotting.QuadrantPlotter import QuadrantPlotter
from src.core.Validation.QuadrantAnalyzer import (
    QuadrantAnalyzer,
    QuadrantMetrics,
    RegionalQuadrantResult,
)


class TestQuadrantAnalysis(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output_dir = pathlib.Path(self.temp_dir.name)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_quadrant_metrics_calculation(self) -> None:
        metrics = QuadrantMetrics(
            both_correct=50,
            model1_only_correct=20,
            model2_only_correct=10,
            both_wrong=20,
            total_events=100,
        )
        self.assertEqual(metrics.both_correct_ratio, 0.5)
        self.assertEqual(metrics.model1_only_correct_ratio, 0.2)
        self.assertEqual(metrics.model2_only_correct_ratio, 0.1)
        self.assertEqual(metrics.both_wrong_ratio, 0.2)

        # McNemar statistic: (|20 - 10| - 1)^2 / (20 + 10) = 81 / 30 = 2.7
        self.assertAlmostEqual(metrics.mcnemar_statistic, 2.7, places=4)
        self.assertGreater(metrics.mcnemar_p_value, 0.0)
        self.assertLessEqual(metrics.mcnemar_p_value, 1.0)

        matrix = metrics.to_matrix()
        self.assertEqual(matrix.shape, (2, 2))
        self.assertEqual(matrix[0, 0], 50)
        self.assertEqual(matrix[0, 1], 20)
        self.assertEqual(matrix[1, 0], 10)
        self.assertEqual(matrix[1, 1], 20)

    def test_quadrant_metrics_zero_discordance(self) -> None:
        metrics = QuadrantMetrics(
            both_correct=80,
            model1_only_correct=0,
            model2_only_correct=0,
            both_wrong=20,
            total_events=100,
        )
        self.assertEqual(metrics.mcnemar_statistic, 0.0)
        self.assertEqual(metrics.mcnemar_p_value, 1.0)

    def test_compute_quadrant_metrics_from_predictions(self) -> None:
        y_true = np.array([1, 1, 0, 0, 1])
        pred1 = np.array([1, 1, 0, 1, 0])
        pred2 = np.array([1, 0, 0, 0, 0])

        metrics = QuadrantAnalyzer.compute_quadrant_metrics(pred1, pred2, y_true)
        # Event 0: y=1, m1=1 (R), m2=1 (R) -> Both Correct
        # Event 1: y=1, m1=1 (R), m2=0 (W) -> M1 Only Correct
        # Event 2: y=0, m1=0 (R), m2=0 (R) -> Both Correct
        # Event 3: y=0, m1=1 (W), m2=0 (R) -> M2 Only Correct
        # Event 4: y=1, m1=0 (W), m2=0 (W) -> Both Wrong
        self.assertEqual(metrics.both_correct, 2)
        self.assertEqual(metrics.model1_only_correct, 1)
        self.assertEqual(metrics.model2_only_correct, 1)
        self.assertEqual(metrics.both_wrong, 1)
        self.assertEqual(metrics.total_events, 5)

    def _create_mock_result(self, iet: int = 1, ieta: int = 1) -> RegionalQuadrantResult:
        metrics = QuadrantMetrics(40, 10, 5, 5, 60)
        sig_metrics = QuadrantMetrics(25, 5, 3, 2, 35)
        bg_metrics = QuadrantMetrics(15, 5, 2, 3, 25)
        probs1 = np.random.uniform(0, 1, 60)
        probs2 = np.random.uniform(0, 1, 60)
        labels = np.array([1] * 35 + [0] * 25)
        et = np.random.uniform(15.0, 100.0, 60)
        eta = np.random.uniform(0.0, 2.5, 60)
        return RegionalQuadrantResult(
            iet=iet,
            ieta=ieta,
            model1_name="ModelA",
            model2_name="ModelB",
            overall_metrics=metrics,
            signal_metrics=sig_metrics,
            background_metrics=bg_metrics,
            probs_model1=probs1,
            probs_model2=probs2,
            labels=labels,
            threshold_model1=0.5,
            threshold_model2=0.5,
            et=et,
            eta=eta,
        )

    def test_quadrant_plotter_renders_files(self) -> None:
        plotter = QuadrantPlotter()
        res1 = self._create_mock_result(iet=1, ieta=1)
        res2 = self._create_mock_result(iet=2, ieta=2)

        global_res = self._create_mock_result(iet=-1, ieta=-1)
        saved = plotter.plot([res1, res2, global_res], self.output_dir, file_format="png")
        self.assertEqual(len(saved["score_scatters"]), 2)
        self.assertFalse(any("global" in p for p in saved["score_scatters"]))
        self.assertEqual(len(saved["summary"]), 1)

        for path_list in saved.values():
            for p in path_list:
                self.assertTrue(os.path.exists(p))

    def test_global_result_aggregation(self) -> None:
        res1 = self._create_mock_result(iet=1, ieta=1)
        res2 = self._create_mock_result(iet=2, ieta=2)

        with patch.object(QuadrantAnalyzer, "_load_config", return_value=MagicMock()):
            with patch("src.core.Validation.QuadrantAnalyzer.NeuralRingerTrainer"):
                with patch("src.core.Validation.QuadrantAnalyzer.SplitManifest"):
                    with patch("src.core.Validation.QuadrantAnalyzer.ResultAggregator"):
                        analyzer = QuadrantAnalyzer(
                            config_path1="config/dummy.yaml",
                            data_path1=self.output_dir,
                            config_path2="config/dummy.yaml",
                            data_path2=self.output_dir,
                        )
                        global_res = analyzer.compute_global_result([res1, res2])
                        self.assertIsNotNone(global_res)
                        self.assertEqual(global_res.iet, -1)
                        self.assertEqual(global_res.ieta, -1)
                        self.assertEqual(global_res.overall_metrics.total_events, 120)
                        self.assertEqual(global_res.overall_metrics.both_correct, 80)
                        self.assertEqual(len(global_res.probs_model1), 120)
                        self.assertEqual(len(global_res.et), 120)
                        self.assertEqual(len(global_res.eta), 120)

    def test_skips_region_when_model_missing(self) -> None:
        with patch.object(QuadrantAnalyzer, "_load_config", return_value=MagicMock()):
            with patch("src.core.Validation.QuadrantAnalyzer.NeuralRingerTrainer") as mock_trainer_cls:
                mock_trainer = MagicMock()
                mock_trainer_cls.return_value = mock_trainer
                mock_trainer.full_dataset.file_paths = ["consolidated.et1.eta1.npz"]
                mock_trainer.full_dataset.__len__.return_value = 1

                with patch("src.core.Validation.QuadrantAnalyzer.SplitManifest"):
                    with patch("src.core.Validation.QuadrantAnalyzer.ResultAggregator") as mock_agg_cls:
                        mock_agg1 = MagicMock()
                        mock_agg2 = MagicMock()
                        mock_agg_cls.side_effect = [mock_agg1, mock_agg2]

                        # Case 1: Model 1 has region, Model 2 does not
                        mock_agg1.has_region.return_value = True
                        mock_agg2.has_region.return_value = False

                        analyzer = QuadrantAnalyzer(
                            config_path1="config/dummy.yaml",
                            data_path1=self.output_dir,
                            config_path2="config/dummy.yaml",
                            data_path2=self.output_dir,
                        )
                        result = analyzer.analyze_region(0)
                        self.assertIsNone(result)

                        # Case 2: Model 1 does not have region, Model 2 has region
                        mock_agg1.has_region.return_value = False
                        mock_agg2.has_region.return_value = True
                        result2 = analyzer.analyze_region(0)
                        self.assertIsNone(result2)

    def test_cli_argument_parsing(self) -> None:
        from scripts.run_quadrant_analysis import parser

        args = parser.parse_args(
            [
                "--config1",
                "config/ModelV1.yaml",
                "--data_path1",
                "results/run1",
                "--data_path2",
                "results/run2",
                "--threshold_mode",
                "calibrated",
                "--output_dir",
                "Plots/quandrantic_analysis",
            ]
        )
        self.assertEqual(args.config1, "config/ModelV1.yaml")
        self.assertIsNone(args.config2)
        self.assertEqual(args.threshold_mode, "calibrated")
        self.assertEqual(args.output_dir, "Plots/quandrantic_analysis")

    def test_comparison_tag_and_output_dir(self) -> None:
        cfg1 = MagicMock()
        cfg1.model.object_name = "ModelV5"
        cfg1.model.parameters = {"input_dim": 100}
        cfg1.config_name = "ModelV5_100Rings"

        cfg2 = MagicMock()
        cfg2.model.object_name = "ModelV6"
        cfg2.model.parameters = {"input_dim": 100}
        cfg2.config_name = "ModelV6_100Rings"

        with patch.object(QuadrantAnalyzer, "_load_config", side_effect=[cfg1, cfg2]):
            with patch("src.core.Validation.QuadrantAnalyzer.NeuralRingerTrainer"):
                with patch("src.core.Validation.QuadrantAnalyzer.SplitManifest"):
                    with patch("src.core.Validation.QuadrantAnalyzer.ResultAggregator"):
                        analyzer = QuadrantAnalyzer("c1.yaml", "d1", "c2.yaml", "d2")
                        self.assertEqual(analyzer.comparison_tag, "ModelV5_100rings_vs_ModelV6_100rings")

                        out_dir = analyzer.get_output_dir("Plots/quandrantic_analysis")
                        self.assertEqual(
                            out_dir,
                            pathlib.Path("Plots/quandrantic_analysis/ModelV5_100rings_vs_ModelV6_100rings"),
                        )
    def test_different_model_dataset_input_shapes(self) -> None:
        import torch
        import torch.nn as nn
        from src.core.Validation.HoldoutEvaluator import HoldoutEvaluator

        cfg1 = MagicMock()
        cfg1.config_name = "Model100"
        cfg1.model.object_name = "Model100"
        cfg1.model.parameters = {"input_dim": 100}

        cfg2 = MagicMock()
        cfg2.config_name = "Model25"
        cfg2.model.object_name = "Model25"
        cfg2.model.parameters = {"input_dim": 25}

        class MockNet100(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.fc = nn.Linear(100, 1)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return torch.sigmoid(self.fc(x))

        class MockNet25(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.fc = nn.Linear(25, 1)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return torch.sigmoid(self.fc(x))

        with patch.object(QuadrantAnalyzer, "_load_config", side_effect=[cfg1, cfg2]):
            with patch("src.core.Validation.QuadrantAnalyzer.NeuralRingerTrainer") as mock_trainer_cls:
                mock_trainer1 = MagicMock()
                mock_trainer2 = MagicMock()
                mock_trainer_cls.side_effect = [mock_trainer1, mock_trainer2]

                mock_trainer1.device = torch.device("cpu")
                mock_trainer1.full_dataset.file_paths = ["consolidated.et1.eta1.npz"]
                mock_trainer1.full_dataset.ring_column_indices = np.arange(100)
                data1 = np.ones((50, 181), dtype=np.float32)
                target1 = np.array([1, 0] * 25, dtype=np.int32)
                mock_trainer1.full_dataset.__getitem__.return_value = (data1, target1, "consolidated.et1.eta1.npz")
                mock_trainer1.factory.create_model.return_value = MockNet100()

                mock_trainer2.device = torch.device("cpu")
                mock_trainer2.full_dataset.file_paths = ["consolidated.et1.eta1.npz"]
                mock_trainer2.full_dataset.ring_column_indices = np.arange(25)
                data2 = np.ones((50, 50), dtype=np.float32)
                target2 = np.array([1, 0] * 25, dtype=np.int32)
                mock_trainer2.full_dataset.__getitem__.return_value = (data2, target2, "consolidated.et1.eta1.npz")
                mock_trainer2.factory.create_model.return_value = MockNet25()

                with patch("src.core.Validation.QuadrantAnalyzer.SplitManifest") as mock_manifest_cls:
                    mock_manifest = MagicMock()
                    mock_manifest.data = {"et_1_eta_1": {}}
                    mock_manifest.get_region_splits.return_value = (np.arange(10, dtype=np.int64), [])
                    mock_manifest_cls.return_value = mock_manifest

                    with patch("src.core.Validation.QuadrantAnalyzer.ResultAggregator") as mock_agg_cls:
                        mock_agg1 = MagicMock()
                        mock_agg2 = MagicMock()
                        mock_agg_cls.side_effect = [mock_agg1, mock_agg2]

                        mock_agg1.has_region.return_value = True
                        mock_agg2.has_region.return_value = True
                        mock_agg1.get_best_model_for_region.return_value = ({"best_weights": MockNet100().state_dict()}, 0, 0.9, 0.01)
                        mock_agg2.get_best_model_for_region.return_value = ({"best_weights": MockNet25().state_dict()}, 0, 0.88, 0.01)

                        analyzer = QuadrantAnalyzer("c1.yaml", "d1", "c2.yaml", "d2", threshold_mode="default")
                        result = analyzer.analyze_region((1, 1))

                        self.assertIsNotNone(result)
                        self.assertEqual(result.overall_metrics.total_events, 10)
                        self.assertEqual(len(result.probs_model1), 10)
                        self.assertEqual(len(result.probs_model2), 10)

    def test_holdout_evaluator_raises_when_ring_indices_none(self) -> None:
        from src.core.Validation.HoldoutEvaluator import HoldoutEvaluator
        evaluator = HoldoutEvaluator()
        with self.assertRaises(ValueError):
            evaluator.evaluate(
                model=MagicMock(),
                data=np.ones((10, 10)),
                target=np.ones(10),
                test_indices=np.arange(5),
                ring_column_indices=None,
                device="cpu",
            )


if __name__ == "__main__":
    unittest.main()


