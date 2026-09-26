from typing import Any, List, Optional
import matplotlib.pyplot as plt
import numpy as np

from src.core.Plotting.BasePlotter import BasePlotter
from src.core.Plotting.Context import RegionPlotContext
from src.utils import norm1


class ProfileMeanEnergyPlotter(BasePlotter):
    """Generates average normalized energy profile curves across calorimeter subdetector rings."""

    def __init__(self) -> None:
        """Initializes the ProfileMeanEnergyPlotter."""
        super().__init__(name="ProfileMeanEnergyPlotter")

    def plot(
        self,
        context: RegionPlotContext,
        x_rings: Optional[np.ndarray] = None,
        plot_name: str = "RingsMeanProfiles_NeuralRinger",
        plot_title: str = "Average Energy Profile in the Rings - NeuralRinger",
        **kwargs: Any,
    ) -> str:
        """Renders and saves the mean energy profile comparing signal vs background.

        Args:
            context: RegionPlotContext with kinematic region data.
            x_rings: Optional explicit ring array (defaults to context.x_holdout_rings).
            plot_name: Suffix string for output PDF naming.
            plot_title: Chart title text.
            **kwargs: Extra plotting parameters.

        Returns:
            Absolute file path of the generated PDF figure.
        """
        rings_data = x_rings if x_rings is not None else context.x_holdout_rings
        y_holdout = context.y_holdout
        ring_index = context.ring_indexes

        normalized_rings = norm1(rings_data)
        signal_data = normalized_rings[y_holdout == 1]
        bg_data = normalized_rings[y_holdout == 0]

        n_rings = signal_data.shape[1]
        x_axis = np.arange(n_rings)

        mean_signal = np.mean(signal_data, axis=0)
        std_signal = np.std(signal_data, axis=0)
        mean_bkg = np.mean(bg_data, axis=0)
        std_bkg = np.std(bg_data, axis=0)
        y_max = max(float(np.max(mean_bkg)), float(np.max(mean_signal)))

        def get_ring_pos(val: int) -> int:
            if ring_index is None:
                return 0
            idx = np.where(np.atleast_1d(ring_index) == val)[0]
            return int(idx[0]) if len(idx) > 0 else 0

        subdet_names: List[str] = ["PreSampler", "EM1", "EM2", "EM3", "TileCal"]
        subdet_x: List[int] = [
            0,
            get_ring_pos(8),
            get_ring_pos(72),
            get_ring_pos(80),
            get_ring_pos(88),
        ]
        subdet_colors: List[str] = ["#1b9e77", "#d95f09", "#7570b3", "#e7298a", "#66a61e"]

        fig, ax = plt.subplots(figsize=(10, 5), clear=True, num=1)

        ax.errorbar(
            x_axis,
            mean_signal,
            yerr=std_signal,
            marker="o",
            mfc="navy",
            mec="navy",
            ms=3,
            mew=0.5,
            elinewidth=0.8,
            capsize=2,
            ecolor="navy",
            color="navy",
            label="Photons",
        )
        ax.errorbar(
            x_axis,
            mean_bkg,
            yerr=std_bkg,
            marker="s",
            mfc="darkorange",
            mec="darkorange",
            ms=3,
            mew=0.5,
            elinewidth=0.8,
            capsize=2,
            ecolor="darkorange",
            color="darkorange",
            label="Hadronic Jets",
        )

        for x_pos, name, color in zip(subdet_x, subdet_names, subdet_colors):
            ax.axvline(x=x_pos, color=color, linestyle="--", linewidth=1)
            ax.text(
                x_pos + 1.2,
                y_max * 1.2,
                name,
                rotation=90,
                va="bottom",
                ha="center",
                fontsize=9,
                color=color,
            )

        ax.set_xlabel("Rings", fontsize=13)
        ax.set_ylabel("Normalized Energy", fontsize=13)

        ticks_loc = np.linspace(0, n_rings, 10, dtype=int)
        ax.set_xticks(ticks_loc)
        ax.set_xticklabels([str(i + 1) for i in ticks_loc], fontsize=11)
        ax.tick_params(axis="y", labelsize=11)

        ax.set_ylim(-0.05, y_max * 1.5)
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend(fontsize=10, loc="upper right")
        ax.set_title(plot_title, fontsize=14)
        fig.tight_layout()

        filename = f"et{context.iet}.eta{context.ieta}.{plot_name}.pdf"
        return self.save_figure(fig, context.output_dir, "RingsMeanProfiles", filename)
