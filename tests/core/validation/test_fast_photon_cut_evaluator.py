import os
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

from src.core.Validation.FastPhotonCutEvaluator import (
    FastPhotonCutEvaluator,
    RegionEfficiencyAccumulator,
    TrigFastPhotonCutMaps,
    UserKinematicGrid,
)


class TestFastPhotonCutEvaluator(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.feature_names = [
            "trig_L2_calo_et",
            "trig_L2_calo_eta",
            "trig_L2_calo_phi",
            "trig_L2_calo_e237",
            "trig_L2_calo_e277",
            "trig_L2_calo_ehad1",
        ]

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_user_kinematic_grid(self) -> None:
        grid = UserKinematicGrid()
        self.assertEqual(grid.num_et_bins, 5)
        self.assertEqual(grid.num_eta_bins, 6)

        et_gev = np.array([10.0, 16.0, 25.0, 35.0, 45.0, 55.0])
        et_bins = grid.get_et_bin(et_gev)
        np.testing.assert_array_equal(et_bins, np.array([-1, 0, 1, 2, 3, 4]))

        abs_eta = np.array([0.5, 1.0, 1.45, 2.0, 2.45, 2.6])
        eta_bins = grid.get_eta_bin(abs_eta)
        np.testing.assert_array_equal(eta_bins, np.array([0, 1, 2, 3, 4, 5]))

        self.assertEqual(grid.get_athena_threshold_for_et_bin(0), 15.0)
        self.assertEqual(grid.get_athena_threshold_for_et_bin(1), 20.0)
        self.assertEqual(grid.get_athena_threshold_for_et_bin(2), 30.0)
        self.assertEqual(grid.get_athena_threshold_for_et_bin(3), 40.0)
        self.assertEqual(grid.get_athena_threshold_for_et_bin(4), 40.0)

    def test_cut_maps_threshold_ranges(self) -> None:
        maps_15 = TrigFastPhotonCutMaps(15.0)
        self.assertEqual(len(maps_15.maps_car_core_thr["loose"]), 9)
        self.assertEqual(maps_15.maps_car_core_thr["loose"][0], 0.809875)

        maps_20 = TrigFastPhotonCutMaps(20.0)
        self.assertEqual(maps_20.maps_car_core_thr["loose"][0], 0.819375)

        with self.assertRaises(ValueError):
            TrigFastPhotonCutMaps(-1.0)

    def test_accumulator_and_evaluate_files(self) -> None:
        grid = UserKinematicGrid()
        evaluator = FastPhotonCutEvaluator(grid=grid)

        # 2 events:
        # Event 1: ET = 25 GeV (et_bin 1), eta = 0.5 (eta_bin 0), passes loose
        # Event 2: ET = 25 GeV (et_bin 1), eta = 0.5 (eta_bin 0), fails loose
        data = np.array(
            [
                [25000.0, 0.5, 0.0, 900.0, 1000.0, 50.0],
                [25000.0, 0.5, 0.0, 500.0, 1000.0, 50.0],
            ],
            dtype=np.float64,
        )
        feature = np.array(self.feature_names, dtype=object)

        file_path = os.path.join(self.temp_dir.name, "sample.npz")
        np.savez(file_path, data=data, feature=feature)

        df_results = evaluator.evaluate_files([file_path], working_points=["loose"])
        self.assertEqual(len(df_results), grid.num_et_bins * grid.num_eta_bins)

        target_row = df_results[(df_results["et_bin"] == 1) & (df_results["eta_bin"] == 0)]
        self.assertEqual(target_row["total_samples"].values[0], 2)
        self.assertEqual(target_row["passed_loose"].values[0], 1)
        self.assertAlmostEqual(target_row["eff_loose"].values[0], 0.5)
        self.assertAlmostEqual(target_row["eff_sig_loose"].values[0], 0.5)

        csv_out = os.path.join(self.temp_dir.name, "efficiencies.csv")
        evaluator.export_reference_csv(df_results, csv_out)
        self.assertTrue(os.path.exists(csv_out))

        df_loaded = pd.read_csv(csv_out)
        self.assertEqual(len(df_loaded), len(df_results))


if __name__ == "__main__":
    unittest.main()
