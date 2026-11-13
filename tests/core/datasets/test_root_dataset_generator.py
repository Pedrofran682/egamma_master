import tempfile
import unittest
from pathlib import Path
import awkward as ak
import numpy as np

from src.core.Datasets.RootDatasetGenerator import (
    DEFAULT_ETA_BINS,
    DEFAULT_ET_BINS,
    RootDatasetGenerator,
)


class TestRootDatasetGenerator(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output_dir = Path(self.temp_dir.name)

        self.mock_events = ak.Array({
            "trig_L2_calo_eta": [-0.5, 1.0, 1.45, 2.0],
            "trig_L2_calo_et": [18000.0, 25000.0, 35000.0, 45000.0],
            "mc_type": [13, 14, 42, 15],
            "trig_L2_calo_rings": [
                [1.0, 2.0],
                [3.0, 4.0, 5.0],
                [6.0],
                [],
            ],
            "ph_et": [18.0, 25.0, 35.0, 45.0],
            "mc_origin": [1, 2, 42, 3],
        })

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_convert_batch_to_numpy(self) -> None:
        generator = RootDatasetGenerator(vector_dim=10)
        (
            batch_data,
            features_list,
            eta_arr,
            et_arr,
            mc_type_arr,
        ) = generator._convert_batch_to_numpy(self.mock_events)

        self.assertEqual(batch_data.shape[0], 4)
        self.assertEqual(batch_data.shape[1], 10 + 5)
        self.assertIn("trig_L2_calo_rings_0", features_list)
        self.assertIn("trig_L2_calo_rings_9", features_list)
        self.assertIn("ph_et", features_list)

        np.testing.assert_array_almost_equal(eta_arr, np.array([0.5, 1.0, 1.45, 2.0]))
        np.testing.assert_array_equal(et_arr, np.array([18000.0, 25000.0, 35000.0, 45000.0]))
        np.testing.assert_array_equal(mc_type_arr, np.array([13, 14, 42, 15]))

    def test_filter_and_save_regions_signal(self) -> None:
        generator = RootDatasetGenerator(
            vector_dim=10,
            eta_bins=[0.0, 0.8, 1.37, 2.5],
            et_bins=[15000.0, 30000.0, 50000.0],
        )
        (
            batch_data,
            features_list,
            eta_arr,
            et_arr,
            mc_type_arr,
        ) = generator._convert_batch_to_numpy(self.mock_events)

        saved = generator._filter_and_save_regions(
            batch_idx=0,
            batch_data=batch_data,
            feature_names=features_list,
            eta_arr=eta_arr,
            et_arr=et_arr,
            mc_type_arr=mc_type_arr,
            file_prefix="test_signal",
            is_perf_jf=False,
            output_dir=self.output_dir,
        )

        self.assertGreater(len(saved), 0)
        first_file = saved[0]
        self.assertTrue(first_file.exists())

        with np.load(first_file) as archive:
            self.assertIn("data", archive.files)
            self.assertIn("target", archive.files)
            self.assertIn("source", archive.files)
            self.assertIn("feature", archive.files)
            self.assertTrue(np.all(archive["target"] == 1))

    def test_filter_and_save_regions_background(self) -> None:
        generator = RootDatasetGenerator(
            vector_dim=10,
            eta_bins=[0.0, 2.5],
            et_bins=[15000.0, 50000.0],
        )
        (
            batch_data,
            features_list,
            eta_arr,
            et_arr,
            mc_type_arr,
        ) = generator._convert_batch_to_numpy(self.mock_events)

        saved = generator._filter_and_save_regions(
            batch_idx=0,
            batch_data=batch_data,
            feature_names=features_list,
            eta_arr=eta_arr,
            et_arr=et_arr,
            mc_type_arr=mc_type_arr,
            file_prefix="test_perf_JF17",
            is_perf_jf=True,
            output_dir=self.output_dir,
        )

        self.assertEqual(len(saved), 1)
        with np.load(saved[0]) as archive:
            self.assertEqual(archive["data"].shape[0], 1)
            self.assertEqual(archive["target"][0], 0)
            self.assertEqual(str(archive["source"][0]), "test_perf_JF17")


if __name__ == "__main__":
    unittest.main()
