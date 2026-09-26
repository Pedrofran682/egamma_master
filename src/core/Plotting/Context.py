import pathlib
from dataclasses import dataclass
from typing import Any
import numpy as np
import torch.nn as nn


@dataclass
class RegionPlotContext:
    """Context object carrying kinematic data and model references for plotting.

    Attributes:
        iet: Transverse energy bin index.
        ieta: Pseudorapidity bin index.
        output_dir: Base directory path where figures will be saved.
        data: Full feature matrix for the region.
        target: Binary label vector for the region.
        test_indices: Sample indices corresponding to the holdout test set.
        trainer: Optional reference to NeuralRingerTrainer.
        model: Optional trained PyTorch model instance.
    """

    iet: int
    ieta: int
    output_dir: pathlib.Path | str
    data: np.ndarray
    target: np.ndarray
    test_indices: np.ndarray | None = None
    trainer: Any = None
    model: nn.Module | None = None

    @property
    def x_holdout_full(self) -> np.ndarray:
        """Full feature matrix filtered by test holdout indices."""
        if self.test_indices is None:
            return self.data
        return self.data[self.test_indices]

    @property
    def y_holdout(self) -> np.ndarray:
        """Binary target vector filtered by test holdout indices."""
        if self.test_indices is None:
            return self.target
        return self.target[self.test_indices]

    @property
    def x_holdout_rings(self) -> np.ndarray:
        """Calorimeter ring feature columns filtered by test holdout indices."""
        if self.trainer and hasattr(self.trainer, "full_dataset"):
            return self.x_holdout_full[:, self.trainer.full_dataset.ring_column_indices]
        return self.x_holdout_full

    @property
    def ring_indexes(self) -> np.ndarray | None:
        """Active relative subdetector ring indices from the dataset."""
        if self.trainer and hasattr(self.trainer, "full_dataset"):
            return self.trainer.full_dataset.indexes
        return None
