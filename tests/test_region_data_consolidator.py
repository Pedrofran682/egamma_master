import tempfile
import unittest
from pathlib import Path
import numpy as np

from src.core.Datasets.RegionDataConsolidator import RegionDataConsolidator
from scripts.merge_regions_data import concatenate_npz_by_group


class TestRegionDataConsolidator(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.base_dir = Path(self.temp_dir.name) / "input"
        self.output_dir = Path(self.temp_dir.name) / "output"
        self.base_dir.mkdir(parents=True)
        self.output_dir.mkdir(parents=True)

        self.features = np.array(["trig_L2_calo_rings_0", "mc_type", "mc_origin"])

        self._create_dummy_npz(
            self.base_dir / "sample_part0.et0.eta0.npz",
            data=np.array([[1.0, 13.0, 1.0], [2.0, 14.0, 2.0]]),
            source=np.array(["mc23_part0", "mc23_part0"]),
            target=np.array([1, 1]),
            features=self.features,
        )
        self._create_dummy_npz(
            self.base_dir / "sample_part1.et0.eta0.npz",
            data=np.array([[3.0, 42.0, 42.0]]),
            source=np.array(["mc23_part1"]),
            target=np.array([0]),
        )
        self._create_dummy_npz(
            self.base_dir / "sample_part0.et1.eta2.npz",
            data=np.array([[4.0, 15.0, 3.0]]),
            source=np.array(["mc23_part0"]),
            target=np.array([1]),
            features=self.features,
        )
        self._create_dummy_npz(
            self.base_dir / "sample_ignored.sys.v1.et0.eta0.npz",
            data=np.array([[99.0, 99.0, 99.0]]),
        )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _create_dummy_npz(
        self,
        path: Path,
        data: np.ndarray,
        source: np.ndarray | None = None,
        target: np.ndarray | None = None,
        features: np.ndarray | None = None,
    ) -> None:
        save_kwargs = {"data": data}
        if source is not None:
            save_kwargs["source"] = source
        if target is not None:
            save_kwargs["target"] = target
        if features is not None:
            save_kwargs["feature"] = features
        np.savez(path, **save_kwargs)

    def test_discover_groups(self) -> None:
        consolidator = RegionDataConsolidator(self.base_dir, self.output_dir)
        groups = consolidator.discover_groups()

        self.assertIn(("0", "0"), groups)
        self.assertIn(("1", "2"), groups)
        self.assertEqual(len(groups[("0", "0")]), 2)
        self.assertEqual(len(groups[("1", "2")]), 1)

    def test_consolidate_group_and_verify_contents(self) -> None:
        consolidator = RegionDataConsolidator(self.base_dir, self.output_dir, compressed=False)
        groups = consolidator.discover_groups()

        out_file = consolidator.consolidate_group("0", "0", groups[("0", "0")])
        self.assertIsNotNone(out_file)
        self.assertTrue(out_file.exists())
        self.assertEqual(out_file.name, "consolidated.et0.eta0.npz")

        with np.load(out_file) as archive:
            self.assertIn("data", archive.files)
            self.assertIn("source", archive.files)
            self.assertIn("target", archive.files)
            self.assertIn("feature", archive.files)

            self.assertEqual(archive["data"].shape, (3, 3))
            self.assertEqual(archive["source"].shape, (3,))
            self.assertEqual(archive["target"].shape, (3,))
            np.testing.assert_array_equal(archive["target"], np.array([1, 1, 0]))
            np.testing.assert_array_equal(archive["source"], np.array(["mc23_part0", "mc23_part0", "mc23_part1"]))

    def test_run_multiprocessing(self) -> None:
        consolidator = RegionDataConsolidator(self.base_dir, self.output_dir, max_workers=2)
        outputs = consolidator.run()

        self.assertEqual(len(outputs), 2)
        output_names = {p.name for p in outputs}
        self.assertEqual(output_names, {"consolidated.et0.eta0.npz", "consolidated.et1.eta2.npz"})

    def test_script_wrapper_function(self) -> None:
        outputs = concatenate_npz_by_group(self.base_dir, self.output_dir, max_workers=1)
        self.assertEqual(len(outputs), 2)


if __name__ == "__main__":
    unittest.main()
