import os
import tempfile
import unittest
import numpy as np
import pandas as pd

from src.core.Plotting.Context import RegionPlotContext
from src.core.Plotting.MetricPlotter import BoxplotSPPlotter


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
