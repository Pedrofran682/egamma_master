from typing import Any, Dict, List, Union
import numpy as np
import pandas as pd
import torch

from src.core.Plotting.Context import RegionPlotContext
from src.core.Plotting.LegacyPlotter import SaliencyMapPlotter
from src.core.Plotting.MetricPlotter import (
    BoxplotSPPlotter,
    MetricsGridPlotter,
    ModelMetricsPlotter,
    RocPlotter,
    get_et_axis,
    get_eta_axis,
)
from src.core.Plotting.ProfilePlotter import ProfileMeanEnergyPlotter

__all__ = [
    "plot_profile_mean_energy_rings",
    "plot_boxplot_SP",
    "plot_model_metrics",
    "plot_model_acc",
    "plot_metrics_grid",
    "plot_saliency_comparison_normalized",
    "get_et_axis",
    "get_eta_axis",
]


def plot_profile_mean_energy_rings(
    data: np.ndarray,
    target: np.ndarray,
    ring_index: np.ndarray,
    folder_path: str,
    iet: int = 0,
    ieta: int = 0,
) -> None:
    """Legacy wrapper for ProfileMeanEnergyPlotter generating ring energy profiles.

    Args:
        data: Matrix containing calorimeter ring features.
        target: Binary label vector (signal vs background).
        ring_index: Indices of active rings.
        folder_path: Directory where the output PDF will be saved.
        iet: Transverse energy bin index.
        ieta: Pseudorapidity bin index.
    """
    context = RegionPlotContext(
        iet=iet,
        ieta=ieta,
        output_dir=folder_path,
        data=data,
        target=target,
    )
    plotter = ProfileMeanEnergyPlotter()
    plotter.plot(context, x_rings=data[:, ring_index] if data.ndim > 1 else data)


def plot_boxplot_SP(
    data_type: str,
    all_training_results: List[Dict[str, Union[str, int, object]]],
    folder_path: str,
    iet: int,
    ieta: int,
) -> None:
    """Legacy wrapper for BoxplotSPPlotter displaying metric distributions across folds.

    Args:
        data_type: Metric name to plot (e.g. 'best_sp_value').
        all_training_results: List of fold statistics dictionaries.
        folder_path: Directory where the output PDF will be saved.
        iet: Transverse energy bin index.
        ieta: Pseudorapidity bin index.
    """
    context = RegionPlotContext(
        iet=iet,
        ieta=ieta,
        output_dir=folder_path,
        data=np.array([]),
        target=np.array([]),
    )
    plotter = BoxplotSPPlotter()
    plotter.plot(context, all_training_results=all_training_results, data_type=data_type)


def plot_model_metrics(
    best_model_details: Dict[str, Any],
    folder_path: str,
    iet: int,
    ieta: int,
) -> None:
    """Legacy wrapper for ModelMetricsPlotter displaying loss and accuracy curves.

    Args:
        best_model_details: Training history dictionary with losses and accuracies.
        folder_path: Directory where the output PDF will be saved.
        iet: Transverse energy bin index.
        ieta: Pseudorapidity bin index.
    """
    context = RegionPlotContext(
        iet=iet,
        ieta=ieta,
        output_dir=folder_path,
        data=np.array([]),
        target=np.array([]),
    )
    plotter = ModelMetricsPlotter()
    plotter.plot(context, history_data=best_model_details)


def plot_model_acc(
    best_model_details: Dict[str, Any],
    folder_path: str,
    iet: int,
    ieta: int,
) -> None:
    """Legacy wrapper for RocPlotter displaying the model ROC curve.

    Args:
        best_model_details: Callback metrics dictionary containing 'pd', 'fa', and 'auc_score'.
        folder_path: Directory where the output PDF will be saved.
        iet: Transverse energy bin index.
        ieta: Pseudorapidity bin index.
    """
    context = RegionPlotContext(
        iet=iet,
        ieta=ieta,
        output_dir=folder_path,
        data=np.array([]),
        target=np.array([]),
    )
    plotter = RocPlotter()
    plotter.plot(context, best_model_details=best_model_details)


def plot_metrics_grid(df: pd.DataFrame, model_name: str, folder_path: str) -> None:
    """Legacy wrapper for MetricsGridPlotter displaying metrics on an (ET, eta) grid.

    Args:
        df: DataFrame with metrics aggregated per kinematic bin.
        model_name: Model identifier name.
        folder_path: Directory where the output PNG will be saved.
    """
    context = RegionPlotContext(
        iet=0,
        ieta=0,
        output_dir=folder_path,
        data=np.array([]),
        target=np.array([]),
    )
    plotter = MetricsGridPlotter()
    plotter.plot(context, df=df, model_name=model_name)


def plot_saliency_comparison_normalized(
    model: torch.nn.Module,
    X_test: np.ndarray,
    y_test: np.ndarray,
    folder_path: str,
    iet: int,
    ieta: int,
    n_samples: int = 100,
) -> None:
    """Legacy wrapper for SaliencyMapPlotter displaying gradient saliency profiles.

    Args:
        model: PyTorch neural network.
        X_test: Test features array.
        y_test: Test labels array.
        folder_path: Directory where the output PDF will be saved.
        iet: Transverse energy bin index.
        ieta: Pseudorapidity bin index.
        n_samples: Number of samples per class evaluated.
    """
    context = RegionPlotContext(
        iet=iet,
        ieta=ieta,
        output_dir=folder_path,
        data=X_test,
        target=y_test,
        model=model,
    )
    plotter = SaliencyMapPlotter()
    plotter.plot(context, n_samples=n_samples)
