import pathlib
from dataclasses import dataclass
from typing import Any
import numpy as np
import torch.nn as nn


from src.core.Interfaces.BaseTrainer import BaseTrainer


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
        trainer: Optional reference to model trainer.
        model: Optional trained PyTorch model instance.
    """

    iet: int
    ieta: int
    output_dir: pathlib.Path | str
    data: np.ndarray
    target: np.ndarray
    test_indices: np.ndarray | None = None
    trainer: BaseTrainer | None = None
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
        if self.trainer is not None and self.trainer.full_dataset is not None:
            ring_indices = self.trainer.full_dataset.ring_column_indices
            if ring_indices is not None:
                return self.x_holdout_full[:, ring_indices]
        return self.x_holdout_full

    @property
    def ring_indexes(self) -> np.ndarray | None:
        """Active relative subdetector ring indices from the dataset."""
        if self.trainer is not None and self.trainer.full_dataset is not None:
            return self.trainer.full_dataset.indexes
        return None


@dataclass
class MetricPlotContext:
    """Encapsulates context and data payloads required by metric and evaluation plotters.

    Attributes:
        region_context: Spatial and kinematic region plot context.
        history_data: Training loss, accuracy, and metric epoch histories.
        callback_metrics: Peak operating points, knee threshold, and ROC details.
        all_training_results: Consolidated DataFrame or records across all folds/repeats.
    """

    region_context: RegionPlotContext
    history_data: dict[str, Any] | None = None
    callback_metrics: dict[str, Any] | None = None
    all_training_results: Any = None

