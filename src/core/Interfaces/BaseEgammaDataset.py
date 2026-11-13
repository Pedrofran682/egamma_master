import logging
import os
from abc import ABC, abstractmethod
from functools import lru_cache
from typing import Any, Dict, List, Tuple
import numpy as np
import numpy.typing as npt
from torch.utils.data import Dataset

from src.utils import norm1

log = logging.getLogger()


class BaseEgammaDataset(Dataset, ABC):
    """Abstract base class for egamma calorimeter NPZ datasets.

    Attributes:
        file_paths: List of file paths matching startswith and endswith filters.
        percentage: Fraction of calorimeter rings selected per subdetector.
        indexes: Subdetector relative indices of the active calorimeter rings.
        percentage_dim: Total count of active ring features.
        ring_column_indices: Absolute column indices in the dataset matrix corresponding to rings.
        config: Optional training configuration object.
    """

    def __init__(
        self,
        drive_path: str,
        startswith: str = "consolidated.et",
        endswith: str = ".npz",
        percentage: float = 1.0,
    ) -> None:
        """Initializes BaseEgammaDataset.

        Args:
            drive_path: Directory path where NPZ files are located.
            startswith: Prefix filter for file discovery.
            endswith: Suffix filter for file discovery.
            percentage: Percentage of calorimeter rings to retain (0.0 to 1.0).
        """
        self.file_paths: List[str] = self._get_files_paths(drive_path, startswith, endswith)
        self.percentage: float = percentage
        self.indexes: List[int] = self._get_rings_index(self.percentage)
        self.percentage_dim: int = len(self.indexes)
        self.ring_column_indices: npt.NDArray[np.int64] | None = None
        self.config: Any = None
        self._feature_names: npt.NDArray[Any] | None = None

    @property
    def feature_names(self) -> npt.NDArray[Any]:
        """Returns the array of feature column names from the dataset.

        Returns:
            Array of feature name strings.

        Raises:
            FileNotFoundError: If no dataset files are available to extract feature names.
        """
        if self._feature_names is None:
            if not self.file_paths:
                raise FileNotFoundError("No files available to extract feature names.")
            with np.load(self.file_paths[0], allow_pickle=True) as sample:
                self._feature_names = np.array(sample["feature"])
        return self._feature_names

    def __len__(self) -> int:
        """Returns total count of discovered NPZ dataset archives."""
        return len(self.file_paths)

    def __getitem__(self, idx: int) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.int32], str]:
        """Loads and processes samples for a given file index.

        Args:
            idx: Index of file within file_paths.

        Returns:
            Tuple of (features, labels, file_path).
        """
        with np.load(self.file_paths[idx], allow_pickle=True) as samples:
            features, labels = self._create_rings(samples)
        return features, labels, self.file_paths[idx]

    def _set_ring_column_indices(self, features: npt.NDArray[Any]) -> None:
        """Determines and sets column indices corresponding to calorimeter rings.

        Args:
            features: Feature name array from the NPZ samples dictionary.
        """
        self._feature_names = features
        first_ring_index = np.where(features == "trig_L2_calo_rings_0")[0][0]
        ring_indices = np.arange(first_ring_index, first_ring_index + 100)
        self.ring_column_indices = ring_indices[self.indexes]

    @abstractmethod
    def filter_events(
        self, samples: Dict[str, Any]
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.int32]]:
        """Filters signal and background events from raw sample archives.

        Args:
            samples: Raw dictionary loaded from the NPZ archive.

        Returns:
            Tuple of (filtered_dataset_matrix, binary_labels_vector).
        """
        pass

    def _filter_events(
        self, samples: Dict[str, Any]
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.int32]]:
        """Backward-compatible alias delegating to public filter_events method."""
        return self.filter_events(samples)

    def _create_rings(
        self, samples: Dict[str, Any]
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.int32]]:
        """Filters events and normalizes selected calorimeter ring features.

        Args:
            samples: Raw dictionary loaded from the NPZ archive.

        Returns:
            Tuple of (normalized_dataset_matrix, binary_labels_vector).
        """
        data_full, y = self.filter_events(samples)
        if self.ring_column_indices is None:
            self._set_ring_column_indices(samples["feature"])

        data_full[:, self.ring_column_indices] = norm1(data_full[:, self.ring_column_indices])
        return data_full, y

    @lru_cache(maxsize=None)
    def _get_rings_index(self, percentage: float) -> List[int]:
        """Calculates active ring feature indices across each calorimeter subdetector.

        Args:
            percentage: Retention ratio per subdetector layer.

        Returns:
            Combined list of active ring indices.
        """
        presampler = list(range(0, 8))
        em1 = list(range(8, 72))
        em2 = list(range(72, 80))
        em3 = list(range(80, 88))
        tilecal = list(range(88, 100))

        presampler = presampler[: round(len(presampler) * percentage)]
        em1 = em1[: round(len(em1) * percentage)]
        em2 = em2[: round(len(em2) * percentage)]
        em3 = em3[: round(len(em3) * percentage)]
        tilecal = tilecal[: round(len(tilecal) * percentage)]

        return presampler + em1 + em2 + em3 + tilecal

    def get_model_dim(self) -> int:
        """Returns the input dimensionality for the neural network.

        Returns:
            Number of active ring features.

        Raises:
            ValueError: If dimension was not properly computed.
        """
        if self.percentage_dim is not None:
            return self.percentage_dim
        raise ValueError("Dimension was not set.")

    def _get_files_paths(self, drive_path: str, startswith: str, endswith: str) -> List[str]:
        """Scans directory and returns list of paths matching filters.

        Args:
            drive_path: Target directory.
            startswith: Filename prefix.
            endswith: Filename extension.

        Returns:
            List of matching absolute file paths.
        """
        return [
            os.path.join(drive_path, file)
            for file in os.listdir(drive_path)
            if file.endswith(endswith) and file.startswith(startswith)
        ]
