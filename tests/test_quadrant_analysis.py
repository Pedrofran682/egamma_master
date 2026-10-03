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
        )

    def test_quadrant_plotter_renders_files(self) -> None:
        plotter = QuadrantPlotter()
        res1 = self._create_mock_result(iet=1, ieta=1)
        res2 = self._create_mock_result(iet=2, ieta=2)

        saved = plotter.plot([res1, res2], self.output_dir, file_format="png")
        self.assertEqual(len(saved["matrix_heatmaps"]), 2)
        self.assertEqual(len(saved["score_scatters"]), 2)
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


if __name__ == "__main__":
    unittest.main()

