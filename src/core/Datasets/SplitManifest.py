import json
import logging
import os
from typing import Any, Dict, List, Tuple
import numpy as np

log = logging.getLogger(__name__)


class SplitManifest:
    """Manages persistent train, validation, and test dataset partitions.

    Attributes:
        manifest_file: File system path where the split manifest JSON is stored.
        data: Internal dictionary holding region partition metadata.
    """

    def __init__(self, manifest_file: str) -> None:
        """Initializes the SplitManifest manager.

        Args:
            manifest_file: Path to the JSON manifest file.
        """
        self.manifest_file: str = manifest_file
        self.data: Dict[str, Any] = {}

    def exists(self) -> bool:
        """Checks if the manifest file exists on disk.

        Returns:
            True if the manifest file exists, False otherwise.
        """
        return os.path.exists(self.manifest_file)

    def load(self) -> Dict[str, Any]:
        """Loads the partition data from the JSON manifest file.

        Returns:
            Dictionary containing the loaded split data.
        """
        with open(self.manifest_file, "r") as f:
            self.data = json.load(f)
        return self.data

    def save(self, data: Dict[str, Any] | None = None) -> None:
        """Serializes partition data to disk as a formatted JSON document.

        Args:
            data: Optional data dictionary to update before saving.
        """
        if data is not None:
            self.data = data
        os.makedirs(os.path.dirname(self.manifest_file), exist_ok=True)
        with open(self.manifest_file, "w") as f:
            json.dump(self.data, f, indent=4)

    def create_region_splits(
        self,
        region_key: str,
        features: np.ndarray,
        labels: np.ndarray,
        kfold_splitter: Any,
        test_fold_idx: int = 0,
    ) -> Tuple[np.ndarray, List[Tuple[np.ndarray, np.ndarray]]]:
        """Generates cross-validation and holdout test partitions for a kinematic region.

        Args:
            region_key: Unique identifier for the kinematic region (e.g. 'et_0_eta_0').
            features: Input feature matrix.
            labels: Binary class labels array.
            kfold_splitter: K-fold cross-validation splitter instance.
            test_fold_idx: Fold index reserved as the holdout test set.

        Returns:
            A tuple containing:
                - holdout test sample indices array.
                - List of tuples with (train_indices, val_indices) for each CV fold.

        Raises:
            ValueError: If test_fold_idx is outside valid split bounds.
        """
        all_folds = [test_idx.tolist() for _, test_idx in kfold_splitter.split(features, labels)]
        if not (0 <= test_fold_idx < len(all_folds)):
            raise ValueError(f"test_fold_idx must be between 0 and {len(all_folds) - 1}")

        holdout_indices = all_folds[test_fold_idx]
        remaining_folds = [fold for idx, fold in enumerate(all_folds) if idx != test_fold_idx]

        cv_splits = []
        for idx in range(len(remaining_folds)):
            val_indices = remaining_folds[idx]
            train_folds = [f for j, f in enumerate(remaining_folds) if j != idx]
            train_indices = [idx_item for subfold in train_folds for idx_item in subfold]
            cv_splits.append({"train": train_indices, "val": val_indices})

        self.data[region_key] = {
            "test_indices": holdout_indices,
            "cv_splits": cv_splits,
        }
        return np.array(holdout_indices), [
            (np.array(split["train"]), np.array(split["val"])) for split in cv_splits
        ]

    def get_region_splits(
        self, region_key: str
    ) -> Tuple[np.ndarray, List[Tuple[np.ndarray, np.ndarray]]]:
        """Retrieves stored train/val/test partitions for a specified region.

        Args:
            region_key: Unique identifier of the requested region.

        Returns:
            A tuple containing:
                - holdout test sample indices array.
                - List of (train_indices, val_indices) numpy arrays for CV folds.

        Raises:
            KeyError: If the region key is not present in the manifest.
        """
        if region_key not in self.data:
            raise KeyError(f"Region key {region_key} not found in split manifest.")

        region_data = self.data[region_key]
        test_indices = np.array(region_data["test_indices"])
        cv_splits = [
            (np.array(split["train"]), np.array(split["val"]))
            for split in region_data["cv_splits"]
        ]
        return test_indices, cv_splits
