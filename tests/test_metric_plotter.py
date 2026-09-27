import os
import tempfile
import unittest
import numpy as np
import pandas as pd

from src.core.Plotting.Context import RegionPlotContext
from src.core.Plotting.MetricPlotter import BoxplotSPPlotter


from scripts.run_plots import PlotterRunner
from scripts.run_roc_plots import RocPlotRunner


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
