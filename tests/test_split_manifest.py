import tempfile
import unittest
import numpy as np
from sklearn.model_selection import StratifiedKFold

from src.core.Datasets.SplitManifest import SplitManifest


class TestSplitManifest(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.features = np.random.randn(100, 10).astype(np.float32)
        self.labels = np.random.choice([0, 1], size=100).astype(np.int32)
        self.kfold_splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_creates_and_persists_splits(self):
        manifest_file = f"{self.temp_dir.name}/splits.json"
        manifest = SplitManifest(manifest_file)

        self.assertFalse(manifest.exists())

        test_indices, cv_splits = manifest.create_region_splits(
            region_key="et_0_eta_0",
            features=self.features,
            labels=self.labels,
            kfold_splitter=self.kfold_splitter,
            test_fold_idx=0,
        )
        manifest.save()

        self.assertTrue(manifest.exists())
        self.assertEqual(len(test_indices), 20)
        self.assertEqual(len(cv_splits), 4)
        self.assertEqual(len(cv_splits[0][0]), 60)
        self.assertEqual(len(cv_splits[0][1]), 20)

    def test_reloads_persisted_splits_deterministically(self):
        manifest_file = f"{self.temp_dir.name}/splits.json"

        creator = SplitManifest(manifest_file)
        created_test, created_cv = creator.create_region_splits(
            region_key="et_1_eta_1",
            features=self.features,
            labels=self.labels,
            kfold_splitter=self.kfold_splitter,
            test_fold_idx=1,
        )
        creator.save()

        loader = SplitManifest(manifest_file)
        self.assertTrue(loader.exists())
        loader.load()

        loaded_test, loaded_cv = loader.get_region_splits("et_1_eta_1")

        np.testing.assert_array_equal(created_test, loaded_test)
        self.assertEqual(len(created_cv), len(loaded_cv))
        for (train_c, val_c), (train_l, val_l) in zip(created_cv, loaded_cv):
            np.testing.assert_array_equal(train_c, train_l)
            np.testing.assert_array_equal(val_c, val_l)

    def test_raises_key_error_for_missing_region(self):
        manifest_file = f"{self.temp_dir.name}/splits.json"
        manifest = SplitManifest(manifest_file)
        manifest.save({"existing_region": {"test_indices": [], "cv_splits": []}})

        with self.assertRaises(KeyError):
            manifest.get_region_splits("non_existent_region")

    def test_raises_value_error_for_invalid_test_fold(self):
        manifest = SplitManifest(f"{self.temp_dir.name}/splits.json")

        with self.assertRaises(ValueError):
            manifest.create_region_splits(
                region_key="region",
                features=self.features,
                labels=self.labels,
                kfold_splitter=self.kfold_splitter,
                test_fold_idx=10,
            )


if __name__ == "__main__":
    unittest.main()
