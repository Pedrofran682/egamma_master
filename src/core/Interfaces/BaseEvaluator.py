from abc import ABC, abstractmethod
from typing import Any
import numpy as np
import torch
import torch.nn as nn


class BaseEvaluator(ABC):
    """Abstract base class for model evaluation on validation or holdout data."""

    @abstractmethod
    def evaluate(
        self,
        model: nn.Module,
        data: np.ndarray,
        target: np.ndarray,
        test_indices: np.ndarray,
        ring_column_indices: np.ndarray,
        device: torch.device,
        target_pd: float | None = None,
    ) -> Any:
        """Performs model evaluation and computes performance metrics.

        Args:
            model: Trained neural network model.
            data: Regional feature matrix.
            target: Regional target labels array.
            test_indices: Indices of events designated as holdout test set.
            ring_column_indices: Column indices corresponding to calorimeter ring inputs.
            device: Active PyTorch computation device.
            target_pd: Target signal detection efficiency.

        Returns:
            Evaluation result object containing predictions and metrics.
        """
        pass
