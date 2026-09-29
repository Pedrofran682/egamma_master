import os
import tempfile
import unittest
import numpy as np
import pandas as pd

from src.core.Plotting.Context import RegionPlotContext
from src.core.Plotting.MetricPlotter import BoxplotSPPlotter


from scripts.run_plots import PlotterRunner
from scripts.run_roc_plots import RocPlotRunner


from src.core.Plotting.PlotManager import PlotManager


class TestMetricPlotter(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.context = RegionPlotContext(
            iet=0,
            ieta=0,
            output_dir=self.temp_dir.name,
            data=np.array([]),
            target=np.array([]),
        )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()
        import shutil

        shutil.rmtree("Plots/model_test_run", ignore_errors=True)
        shutil.rmtree("Plots/ROC_Grid", ignore_errors=True)
        shutil.rmtree("Plots/my_model", ignore_errors=True)
        shutil.rmtree("Plots/ModelV1_ConfigTest", ignore_errors=True)
        shutil.rmtree("Plots/ModelV5_RunName", ignore_errors=True)

    def test_plot_manager_run_metrics_for_region_success(self) -> None:
        manager = PlotManager()
        df = pd.DataFrame(
            [
                {
                    "fold": 0,
                    "reapet": 1,
                    "best_sp_value": 0.82,
                    "history": {
                        "train_loss": [0.5, 0.4],
                        "val_loss": [0.6, 0.5],
                        "train_acc": [0.8, 0.85],
                        "val_acc": [0.75, 0.82],
                        "callbackMetrics": {
                            "pd": [0.8, 0.9],
                            "fa": [0.2, 0.1],
                            "sp": [0.8, 0.85],
                            "auc_score": 0.92,
                            "best_knee_point": 1,
                        },
                    },
                },
                {
                    "fold": 1,
                    "reapet": 1,
                    "best_sp_value": 0.91,
                    "history": {
                        "train_loss": [0.4, 0.3],
                        "val_loss": [0.5, 0.4],
                        "train_acc": [0.85, 0.9],
                        "val_acc": [0.8, 0.88],
                        "callbackMetrics": {
                            "pd": [0.85, 0.95],
                            "fa": [0.15, 0.05],
                            "sp": [0.85, 0.91],
                            "auc_score": 0.96,
                            "best_knee_point": 1,
                        },
                    },
                },
            ]
        )
        success = manager.run_metrics_for_region(
            df, self.temp_dir.name, iet=0, ieta=0
        )
        self.assertTrue(success)

    def test_plot_manager_run_metrics_for_region_empty_skips(self) -> None:
        manager = PlotManager()
        empty_df = pd.DataFrame()
        self.assertFalse(
            manager.run_metrics_for_region(empty_df, self.temp_dir.name, iet=0, ieta=0)
        )

        missing_sp_df = pd.DataFrame([{"fold": 0, "other": 1}])
        self.assertFalse(
            manager.run_metrics_for_region(missing_sp_df, self.temp_dir.name, iet=0, ieta=0)
        )

    def test_plotter_runner_execute_skips_empty_regions(self) -> None:
        results_dir = os.path.join(self.temp_dir.name, "results_run")
        os.makedirs(results_dir, exist_ok=True)

        empty_file = os.path.join(results_dir, "repeat0.fold_idx0.iet0.ieta0.pkl")
        pd.DataFrame().to_pickle(empty_file)

        valid_file = os.path.join(results_dir, "repeat0.fold_idx0.iet1.ieta1.pkl")
        pd.DataFrame(
            [
                {
                    "fold": 0,
                    "reapet": 1,
                    "best_sp_value": 0.88,
                    "history": {
                        "train_loss": [0.5],
                        "val_loss": [0.6],
                        "train_acc": [0.8],
                        "val_acc": [0.75],
                        "callbackMetrics": {
                            "pd": [0.8],
                            "fa": [0.2],
                            "sp": [0.88],
                            "auc_score": 0.94,
                            "best_knee_point": 0,
                        },
                    },
                }
            ]
        ).to_pickle(valid_file)

        runner = PlotterRunner(
            results_path=results_dir,
            output_dir=os.path.join(self.temp_dir.name, "output_plots"),
        )
        runner.execute()
        self.assertTrue(os.path.exists(runner.output_dir))

    def test_plotter_runner_default_output_dir(self) -> None:
        runner = PlotterRunner(results_path="results/model_test_run")
        self.assertEqual(str(runner.output_dir), "Plots/model_test_run")
        self.assertTrue(runner.output_dir.exists())

    def test_plotter_runner_custom_output_dir(self) -> None:
        custom_out = os.path.join(self.temp_dir.name, "CustomPlots")
        runner = PlotterRunner(
            results_path="results/model_test_run", output_dir=custom_out
        )
        self.assertEqual(str(runner.output_dir), custom_out)
    def test_plotter_runner_config_path_output_dir(self) -> None:
        runner = PlotterRunner(
            results_path="results/model_test_run",
            config_path="config/NeuralRinger/ModelV1_ConfigTest.yaml",
        )
        self.assertEqual(str(runner.output_dir), "Plots/model_test_run")
        self.assertTrue(runner.output_dir.exists())

    def test_plotter_runner_preserves_results_folder_name(self) -> None:
        runner = PlotterRunner(
            results_path="results/config_nameModelV5_RunName_id20260928210000"
        )
        self.assertEqual(
            str(runner.output_dir),
            "Plots/config_nameModelV5_RunName_id20260928210000",
        )
        self.assertTrue(runner.output_dir.exists())

    def test_plotter_runner_file_path_output_dir(self) -> None:
        runner = PlotterRunner(
            results_path="results/config_nameModelV5_RunName_id20260928210000/model_et1_eta1.pkl"
        )
        self.assertEqual(
            str(runner.output_dir),
            "Plots/config_nameModelV5_RunName_id20260928210000",
        )
        self.assertTrue(runner.output_dir.exists())

    def test_roc_plot_runner_output_dirs(self) -> None:
        runner_base = RocPlotRunner(results_path="results/")
        self.assertEqual(str(runner_base.output_dir), "Plots/ROC_Grid")
        self.assertTrue(runner_base.output_dir.exists())

        runner_model = RocPlotRunner(results_path="results/my_model")
        self.assertEqual(str(runner_model.output_dir), "Plots/my_model")
        self.assertTrue(runner_model.output_dir.exists())

        custom_out = os.path.join(self.temp_dir.name, "CustomROC")
        runner_custom = RocPlotRunner(
            results_path="results/", output_dir=custom_out
        )
        self.assertEqual(str(runner_custom.output_dir), custom_out)
        self.assertTrue(runner_custom.output_dir.exists())

    def test_boxplot_sp_plotter_with_dataframe(self) -> None:
        plotter = BoxplotSPPlotter()
        df = pd.DataFrame(
            [
                {"fold": 0, "best_sp_value": 0.85},
                {"fold": 0, "best_sp_value": 0.86},
                {"fold": 1, "best_sp_value": 0.88},
                {"fold": 1, "best_sp_value": 0.89},
            ]
        )
        saved_path = plotter.plot(self.context, all_training_results=df)
        self.assertIsNotNone(saved_path)
        self.assertTrue(os.path.exists(saved_path))

    def test_boxplot_sp_plotter_with_list_of_dicts(self) -> None:
        plotter = BoxplotSPPlotter()
        data = [
            {"fold": 0, "best_sp_value": 0.85},
            {"fold": 1, "best_sp_value": 0.88},
        ]
        saved_path = plotter.plot(self.context, all_training_results=data)
        self.assertIsNotNone(saved_path)
        self.assertTrue(os.path.exists(saved_path))

    def test_boxplot_sp_plotter_with_empty_dataframe(self) -> None:
        plotter = BoxplotSPPlotter()
        saved_path = plotter.plot(self.context, all_training_results=pd.DataFrame())
        self.assertIsNone(saved_path)

    def test_boxplot_sp_plotter_with_empty_list(self) -> None:
        plotter = BoxplotSPPlotter()
        saved_path = plotter.plot(self.context, all_training_results=[])
        self.assertIsNone(saved_path)

    def test_boxplot_sp_plotter_with_none(self) -> None:
        plotter = BoxplotSPPlotter()
        saved_path = plotter.plot(self.context, all_training_results=None)
        self.assertIsNone(saved_path)

    def test_boxplot_sp_plotter_with_missing_metric_column(self) -> None:
        plotter = BoxplotSPPlotter()
        df = pd.DataFrame([{"fold": 0, "other_metric": 1.0}])
        saved_path = plotter.plot(self.context, all_training_results=df)
        self.assertIsNone(saved_path)


if __name__ == "__main__":
    unittest.main()
