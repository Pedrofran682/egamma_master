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
    StrategyMetrics,
)


class TestQuadrantAnalysis(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output_dir = pathlib.Path(self.temp_dir.name)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_strategy_metrics_calculation(self) -> None:
        preds = np.array([1, 1, 0, 0, 1])
        labels = np.array([1, 0, 0, 1, 1])
        metrics = StrategyMetrics.from_predictions(preds, labels)

        self.assertEqual(metrics.tp, 2)
        self.assertEqual(metrics.fn, 1)
        self.assertEqual(metrics.tn, 1)
        self.assertEqual(metrics.fp, 1)
        self.assertEqual(metrics.total, 5)

        self.assertAlmostEqual(metrics.pd, 2 / 3, places=4)
        self.assertAlmostEqual(metrics.pf, 1 / 2, places=4)
        self.assertAlmostEqual(metrics.eff, 3 / 5, places=4)
        self.assertGreater(metrics.sp, 0.0)

        self.assertGreater(metrics.pd_uncertainty, 0.0)
        self.assertGreater(metrics.pf_uncertainty, 0.0)
        self.assertGreater(metrics.eff_uncertainty, 0.0)
        self.assertGreater(metrics.sp_uncertainty, 0.0)

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

        self.assertAlmostEqual(metrics.mcnemar_statistic, 2.7, places=4)
        self.assertGreater(metrics.mcnemar_p_value, 0.0)
        self.assertLessEqual(metrics.mcnemar_p_value, 1.0)

        matrix = metrics.to_matrix()
        self.assertEqual(matrix.shape, (2, 2))
        self.assertEqual(matrix[0, 0], 50)
        self.assertEqual(matrix[0, 1], 20)
        self.assertEqual(matrix[1, 0], 10)
        self.assertEqual(matrix[1, 1], 20)

    def test_extract_l2_showershapes(self) -> None:
        feature_names = [
            "trig_L2_calo_weta2",
            "trig_L2_calo_wstot",
            "trig_L2_calo_fracs1",
            "trig_L2_calo_ehad1",
            "trig_L2_calo_emaxs1",
            "trig_L2_calo_e2tsts1",
            "trig_L2_calo_e237",
            "trig_L2_calo_e277",
            "trig_L2_calo_et",
        ]
        features = np.array(
            [
                [0.02, 1.5, 0.4, 500.0, 1000.0, 200.0, 800.0, 1000.0, 20000.0],
                [0.03, 1.8, 0.6, 600.0, 1500.0, 500.0, 900.0, 1000.0, 30000.0],
            ]
        )
        shapes = QuadrantAnalyzer.extract_l2_showershapes(features, feature_names)

        self.assertIn("Rcore", shapes)
        self.assertIn("Rhad", shapes)
        self.assertIn("Eratio", shapes)
        self.assertIn("trig_L2_calo_weta2", shapes)

        self.assertAlmostEqual(shapes["Rcore"][0], 0.8, places=4)
        self.assertAlmostEqual(shapes["Rhad"][0], 500.0 / 20000.0, places=4)
        # Eratio: (1000 - 200) / (1000 + 200) = 800 / 1200 = 2/3
        self.assertAlmostEqual(shapes["Eratio"][0], 2.0 / 3.0, places=4)

    def _create_mock_result(self, iet: int = 1, ieta: int = 1) -> RegionalQuadrantResult:
        metrics = QuadrantMetrics(40, 10, 5, 5, 60)
        sig_metrics = QuadrantMetrics(25, 5, 3, 2, 35)
        bg_metrics = QuadrantMetrics(15, 5, 2, 3, 25)

        s1_metrics = StrategyMetrics(0.9, 0.1, 0.89, 0.9, 30, 2, 23, 5, 60)
        s2_metrics = StrategyMetrics(0.85, 0.12, 0.86, 0.88, 28, 3, 22, 7, 60)

        labels = np.array([1] * 35 + [0] * 25)
        preds1 = np.array([1] * 30 + [0] * 5 + [0] * 23 + [1] * 2)
        preds2 = np.array([1] * 28 + [0] * 7 + [0] * 22 + [1] * 3)

        showershapes = {
            "Rcore": np.random.uniform(0.7, 0.95, 60),
            "Rhad": np.random.uniform(0.01, 0.08, 60),
            "Eratio": np.random.uniform(0.5, 0.99, 60),
            "trig_L2_calo_weta2": np.random.uniform(0.01, 0.03, 60),
        }

        et = np.random.uniform(20.0, 50.0, 60)
        eta = np.random.uniform(0.0, 2.5, 60)

        return RegionalQuadrantResult(
            iet=iet,
            ieta=ieta,
            strategy1_name="ModelA",
            strategy2_name="ModelB",
            metrics_strategy1=s1_metrics,
            metrics_strategy2=s2_metrics,
            overall_metrics=metrics,
            signal_metrics=sig_metrics,
            background_metrics=bg_metrics,
            preds_strategy1=preds1,
            preds_strategy2=preds2,
            labels=labels,
            threshold_strategy1=0.5,
            threshold_strategy2=0.5,
            showershapes=showershapes,
            et=et,
            eta=eta,
        )

    def test_quadrant_plotter_renders_histograms_and_summary(self) -> None:
        plotter = QuadrantPlotter()
        res1 = self._create_mock_result(iet=1, ieta=1)
        res2 = self._create_mock_result(iet=2, ieta=2)
        global_res = self._create_mock_result(iet=-1, ieta=-1)

        saved = plotter.plot([res1, res2, global_res], self.output_dir, file_format="png")
        self.assertEqual(len(saved["summary"]), 1)
        self.assertEqual(len(saved["class_efficiencies"]), 1)
        self.assertEqual(len(saved["histograms"]), 8)
        self.assertEqual(len(saved["efficiency_curves"]), 6)

        for p in saved["histograms"]:
            self.assertTrue(os.path.exists(p))
            self.assertTrue(p.endswith("_quadrant_hist.png"))

        for p in saved["efficiency_curves"]:
            self.assertTrue(os.path.exists(p))
            self.assertTrue("_efficiency_vs_" in p)

        self.assertTrue(any("global_efficiency_vs_et.png" in p for p in saved["efficiency_curves"]))
        self.assertTrue(any("global_efficiency_vs_eta.png" in p for p in saved["efficiency_curves"]))

        for p in saved["summary"]:
            self.assertTrue(os.path.exists(p))
            self.assertTrue(p.endswith("regional_quadrant_summary.png"))

        for p in saved["class_efficiencies"]:
            self.assertTrue(os.path.exists(p))
            self.assertTrue(p.endswith("regional_class_efficiencies.png"))

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
                        self.assertEqual(len(global_res.preds_strategy1), 120)
                        self.assertIn("Rcore", global_res.showershapes)
                        self.assertEqual(len(global_res.showershapes["Rcore"]), 120)

    def test_model_vs_cut_initialization_and_tag(self) -> None:
        cfg = MagicMock()
        cfg.model.object_name = "ModelV1"
        cfg.model.parameters = {"input_dim": 100}
        cfg.config_name = "ModelV1_100Rings"

        with patch.object(QuadrantAnalyzer, "_load_config", return_value=cfg):
            with patch("src.core.Validation.QuadrantAnalyzer.NeuralRingerTrainer"):
                with patch("src.core.Validation.QuadrantAnalyzer.SplitManifest"):
                    with patch("src.core.Validation.QuadrantAnalyzer.ResultAggregator"):
                        analyzer = QuadrantAnalyzer(
                            config_path1="c1.yaml",
                            data_path1="d1",
                            mode="model_vs_cut",
                            working_point="loose",
                        )
                        self.assertEqual(analyzer.mode, "model_vs_cut")
                        self.assertEqual(analyzer.comparison_tag, "ModelV1_100rings_vs_Cut_Loose")

    def test_cli_argument_parsing(self) -> None:
        from scripts.run_quadrant_analysis import parser

        args_cut = parser.parse_args(
            [
                "--mode",
                "model_vs_cut",
                "--config1",
                "config/ModelV1.yaml",
                "--data_path1",
                "results/run1",
                "--working_point",
                "tight",
            ]
        )
        self.assertEqual(args_cut.mode, "model_vs_cut")
        self.assertEqual(args_cut.working_point, "tight")
        self.assertIsNone(args_cut.data_path2)

        args_model = parser.parse_args(
            [
                "--config1",
                "config/ModelV1.yaml",
                "--data_path1",
                "results/run1",
                "--data_path2",
                "results/run2",
            ]
        )
        self.assertEqual(args_model.mode, "model_vs_model")
        self.assertEqual(args_model.data_path2, "results/run2")

    def test_save_results_table(self) -> None:
        import pandas as pd

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
                        csv_path = analyzer.save_results_table([res1, res2], self.output_dir)
                        self.assertTrue(csv_path.exists())

                        df = pd.read_csv(csv_path)
                        self.assertEqual(len(df), 3)  # 2 regions + 1 Global
                        self.assertIn("s1_pd", df.columns)
                        self.assertIn("s1_pd_uncertainty", df.columns)
                        self.assertIn("s1_pf", df.columns)
                        self.assertIn("s1_pf_uncertainty", df.columns)
                        self.assertIn("s1_sp", df.columns)
                        self.assertIn("s1_sp_uncertainty", df.columns)
                        self.assertIn("s1_eff", df.columns)
                        self.assertIn("s1_eff_uncertainty", df.columns)
                        self.assertIn("s2_pd", df.columns)
                        self.assertIn("s2_pd_uncertainty", df.columns)
                        self.assertIn("s2_pf", df.columns)
                        self.assertIn("s2_pf_uncertainty", df.columns)
                        self.assertIn("s2_sp", df.columns)
                        self.assertIn("s2_sp_uncertainty", df.columns)
                        self.assertIn("s2_eff", df.columns)
                        self.assertIn("s2_eff_uncertainty", df.columns)
                        self.assertIn("both_correct", df.columns)
                        self.assertIn("mcnemar_p_value", df.columns)


if __name__ == "__main__":
    unittest.main()
