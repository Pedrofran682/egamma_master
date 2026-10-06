from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple
import pandas as pd


class BaseResultAggregator(ABC):
    """Abstract base class for discovering, grouping, and retrieving regional model results."""

    @abstractmethod
    def group_and_concat(self) -> Dict[str, pd.DataFrame]:
        """Groups and consolidates result records across regions.

        Returns:
            Dictionary mapping region keys to consolidated DataFrames.
        """
        pass

    @abstractmethod
    def has_region(self, et: int, eta: int) -> bool:
        """Checks if result records exist for a specified kinematic region.

        Args:
            et: Transverse energy bin index.
            eta: Pseudorapidity bin index.

        Returns:
            True if records exist for the region, False otherwise.
        """
        pass

    @abstractmethod
    def get_best_model_for_region(
        self, et: int, eta: int
    ) -> Tuple[Dict[str, Any], int, float, float]:
        """Extracts the best model details specifically for a given kinematic region.

        Args:
            et: Transverse energy bin index.
            eta: Pseudorapidity bin index.

        Returns:
            Tuple of (best_model_dict, best_repetition, best_mean_sp, best_std_sp).
        """
        pass
