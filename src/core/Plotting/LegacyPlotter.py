import logging
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.core.Interfaces.BasePlotter import BasePlotter
from src.core.Plotting.Context import RegionPlotContext
from src.core.Plotting.ProfilePlotter import ProfileMeanEnergyPlotter
from src.utils import compute_mean_std_saliency

log = logging.getLogger()


def extract_convolutional_features(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Extract intermediate features through conv1 and conv2 layers if present in the model.

    Args:
        model: PyTorch model containing potential convolutional layers.
        x: Input tensor representing feature rings.

    Returns:
        Processed feature tensor after passing through convolution and ReLU activations.
    """
    input_dim = getattr(model, "input_dim", x.shape[-1])
    x = x.view(-1, 1, input_dim)
    if hasattr(model, "conv1") and hasattr(model, "conv2"):
        x = torch.relu(model.conv1(x))
        x = torch.relu(model.conv2(x))
    return x


class ConvLayerPlotter(BasePlotter):
    """Plotter for visualizing intermediate feature maps from convolutional layers."""

    def __init__(self) -> None:
        """Initialize the ConvLayerPlotter instance."""
        super().__init__(name="ConvLayerPlotter")
        self.profile_plotter: ProfileMeanEnergyPlotter = ProfileMeanEnergyPlotter()

    def plot(
        self,
        context: RegionPlotContext,
        plot_title: str = "Feature Extraction - Conv layer output",
        **kwargs: Any,
    ) -> None:
        """Extract and plot channel-wise convolutional feature maps.

        Args:
            context: Context containing model, trainer, and holdout data.
            plot_title: Title prefix for generated channel plots.
            **kwargs: Additional plotting options.
        """
        if context.model is None or context.trainer is None:
            return

        context.model.eval()
        x_holdout_rings = context.x_holdout_rings
        if x_holdout_rings is None:
            return

        tensor_in = torch.from_numpy(x_holdout_rings).float().to(context.trainer.device)
        with torch.no_grad():
            conv_out_tensor = extract_convolutional_features(context.model, tensor_in)

        conv_out_np = conv_out_tensor.squeeze().cpu().numpy()

        for ch in range(4):
            if conv_out_np.ndim == 3:
                feature_slice = conv_out_np[:, ch, :]
            else:
                feature_slice = conv_out_np

            self.profile_plotter.plot(
                context=context,
                x_rings=feature_slice,
                plot_name=f"ConvFeatureExtraction_plot_ringer.ch{ch}",
                plot_title=f"{plot_title}.ch{ch}",
            )


class SaliencyMapPlotter(BasePlotter):
    """Plotter for generating gradient-based saliency comparison profiles."""

    def __init__(self) -> None:
        """Initialize the SaliencyMapPlotter instance."""
        super().__init__(name="SaliencyMapPlotter")

    def plot(
        self,
        context: RegionPlotContext,
        n_samples: int = 1000,
        **kwargs: Any,
    ) -> Optional[str]:
        """Compute and save gradient saliency profiles comparing signal and background.

        Args:
            context: Region context with holdout data and trained model.
            n_samples: Maximum number of samples to average per class.
            **kwargs: Additional plotting options.

        Returns:
            Saved figure filepath or None if context components are missing.
        """
        if context.model is None or context.trainer is None:
            return None

        y_test = context.y_holdout
        x_holdout = context.x_holdout_rings
        if y_test is None or x_holdout is None:
            return None

        x_test = torch.from_numpy(x_holdout).float().to(context.trainer.device)

        idx_signal = np.where(y_test == 1)[0]
        idx_bg = np.where(y_test == 0)[0]
        np.random.shuffle(idx_signal)
        np.random.shuffle(idx_bg)

        idx_signal = idx_signal[:n_samples]
        idx_bg = idx_bg[:n_samples]

        mean_signal, std_signal = compute_mean_std_saliency(
            x_test[idx_signal].cpu().numpy(), context.model
        )
        mean_bg, std_bg = compute_mean_std_saliency(
            x_test[idx_bg].cpu().numpy(), context.model
        )

        fig, ax = plt.subplots(figsize=(12, 6), clear=True, num=1)
        x_range = np.arange(len(mean_signal))

        ax.plot(mean_signal, label="Signal (mean)", color="blue")
        ax.fill_between(
            x_range,
            mean_signal - std_signal,
            mean_signal + std_signal,
            color="blue",
            alpha=0.3,
            label="Signal ±1σ",
        )
        ax.plot(mean_bg, label="Background (mean)", color="red")
        ax.fill_between(
            x_range,
            mean_bg - std_bg,
            mean_bg + std_bg,
            color="red",
            alpha=0.3,
            label="Background ±1σ",
        )

        ax.set_title(
            f"Comparison of Normalized Mean Saliency Profile\n({n_samples} samples per class)\n({context.iet},{context.ieta})"
        )
        ax.set_xlabel("Input Position - Ring Index")
        n_rings = len(mean_signal)
        ticks_loc = np.linspace(0, n_rings, 10, dtype=int)
        ax.set_xticks(ticks_loc)
        ax.set_xticklabels([str(i + 1) for i in ticks_loc])
        ax.set_ylabel("Normalized Importance (Saliency)")
        ax.legend()
        ax.grid(True, linestyle="--")
        fig.tight_layout()

        filename = (
            f"iet{context.iet}.ieta{context.ieta}_saliency_comparison_normalized_{n_rings}Rings.pdf"
        )
        return self.save_figure(fig, context.output_dir, "saliency", filename)
