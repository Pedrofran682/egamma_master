import os
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
import yaml

from src.core.Plotting.RegionDistributionPlotter import RegionDistributionPlotter
from src.core.Validation.RegionDataDistributionAnalyzer import RegionDataDistributionAnalyzer


class TestRegionDistributionPlotter(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.plotter = RegionDistributionPlotter()
        self.distribution_dataframe = pd.DataFrame(
            [
                {
                    "et": 0,
                    "eta": 0,
                    "background_count": 500,
                    "signal_count": 1000,
                    "total_events": 1500,
                    "signal_fraction": 1000 / 1500,
                },
                {
                    "et": 0,
                    "eta": 1,
                    "background_count": 600,
                    "signal_count": 800,
                    "total_events": 1400,
                    "signal_fraction": 800 / 1400,
                },
                {
                    "et": 1,
                    "eta": 0,
                    "background_count": 300,
                    "signal_count": 1200,
                    "total_events": 1500,
                    "signal_fraction": 1200 / 1500,
                },
                {
                    "et": 1,
                    "eta": 1,
                    "background_count": 400,
                    "signal_count": 900,
                    "total_events": 1300,
                    "signal_fraction": 900 / 1300,
                },
            ]
        )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_plot_generates_all_heatmaps(self) -> None:
        saved_figure_paths = self.plotter.plot(
            region_distribution_dataframe=self.distribution_dataframe,
            output_dir=self.temp_dir.name,
            file_format="png",
            subfolder="test_plots",
        )

        self.assertIn("composite_grid", saved_figure_paths)
        self.assertIn("background_count", saved_figure_paths)
        self.assertIn("signal_count", saved_figure_paths)
        self.assertIn("total_events", saved_figure_paths)
        self.assertIn("signal_fraction", saved_figure_paths)

        for metric_name, figure_path in saved_figure_paths.items():
            self.assertTrue(os.path.exists(figure_path), f"Figure {figure_path} does not exist.")
            self.assertGreater(os.path.getsize(figure_path), 0, f"Figure {figure_path} is empty.")


class TestRegionDataDistributionAnalyzer(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.data_dir = Path(self.temp_dir.name) / "data"
        self.data_dir.mkdir(parents=True, exist_ok=True)

        features = np.array(
            ["mc_type", "mc_origin", "trig_L2_calo_rings_0"]
            + [f"ring_{i}" for i in range(1, 100)]
        )
        source = np.array(["perf_JF17_sample"] * 20)

        data_1_1 = np.zeros((20, len(features)))
        data_1_1[:10, 0] = 14
        data_1_1[:10, 1] = 1
        data_1_1[10:, 0] = 0
        data_1_1[10:, 1] = 42
        np.savez(
            self.data_dir / "consolidated.et1.eta1.npz",
            data=data_1_1,
            target=data_1_1[:, 0],
            source=source,
            feature=features,
        )

        data_1_2 = np.zeros((20, len(features)))
        data_1_2[:15, 0] = 14
        data_1_2[:15, 1] = 1
        data_1_2[15:, 0] = 0
        data_1_2[15:, 1] = 42
        np.savez(
            self.data_dir / "consolidated.et1.eta2.npz",
            data=data_1_2,
            target=data_1_2[:, 0],
            source=source,
            feature=features,
        )

        self.config_path = Path(self.temp_dir.name) / "test_config.yaml"
        config_data = {
            "config_name": "TestDistributionRun",
            "batch_size": 32,
            "epochs": 1,
            "n_initializations": 1,
            "pred_target_limiar": 0.5,
            "et_range_idx": [1],
            "eta_range_idx": [1, 2],
            "loss_function": {
                "module": "torch.nn",
                "object_name": "BCELoss",
            },
            "optimizer_function": {
                "module": "torch.optim",
                "object_name": "Adam",
                "parameters": {"lr": 0.01},
            },
            "kFold": {
                "module": "sklearn.model_selection",
                "object_name": "StratifiedKFold",
                "parameters": {"n_splits": 5, "shuffle": True, "random_state": 42},
            },
            "model": {
                "module": "src.Models.egamma.ModelV1",
                "object_name": "ModelV1",
                "parameters": {"input_dim": 100},
            },
            "dataset": {
                "module": "src.core.Datasets.EgammaNpzDataset",
                "object_name": "EgammaNpzDataset",
                "parameters": {
                    "drive_path": str(self.data_dir),
                    "startswith": "consolidated.et",
                    "endswith": ".npz",
                    "percentage": 1.0,
                },
            },
        }
        with open(self.config_path, "w") as f:
            yaml.dump(config_data, f)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_analyzer_collect_and_run(self) -> None:
        output_dir = Path(self.temp_dir.name) / "output"
        analyzer = RegionDataDistributionAnalyzer(
            config_path=str(self.config_path),
            data_path=str(self.data_dir),
            output_dir=str(output_dir),
        )

        distribution_dataframe = analyzer.collect_distribution()
        self.assertEqual(len(distribution_dataframe), 2)

        row_1_1 = distribution_dataframe[
            (distribution_dataframe["et"] == 1) & (distribution_dataframe["eta"] == 1)
        ].iloc[0]
        self.assertEqual(row_1_1["signal_count"], 10)
        self.assertEqual(row_1_1["background_count"], 10)
        self.assertEqual(row_1_1["total_events"], 20)
        self.assertAlmostEqual(row_1_1["signal_fraction"], 0.5)
        self.assertEqual(row_1_1["et_range"], "20-30 GeV")
        self.assertEqual(row_1_1["eta_range"], "0.80-1.37")

        row_1_2 = distribution_dataframe[
            (distribution_dataframe["et"] == 1) & (distribution_dataframe["eta"] == 2)
        ].iloc[0]
        self.assertEqual(row_1_2["signal_count"], 15)
        self.assertEqual(row_1_2["background_count"], 5)
        self.assertEqual(row_1_2["total_events"], 20)
        self.assertAlmostEqual(row_1_2["signal_fraction"], 0.75)
        self.assertEqual(row_1_2["et_range"], "20-30 GeV")
        self.assertEqual(row_1_2["eta_range"], "1.37-1.54")

        result_dataframe = analyzer.run(file_format="png")
        self.assertEqual(len(result_dataframe), 2)

        csv_summary_file = output_dir / "region_data_distribution.csv"
        self.assertTrue(csv_summary_file.exists())

        grid_image = output_dir / "data_distribution_2d_grid.png"
        self.assertTrue(grid_image.exists())
