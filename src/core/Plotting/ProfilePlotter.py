from pathlib import Path
from typing import Any, List, Optional, Tuple, Union
import matplotlib.pyplot as plt
import mplhep as mh
import numpy as np

from src.core.Interfaces.BasePlotter import BasePlotter
from src.core.Plotting.Localization import PlotTextKey
from src.core.Plotting.Context import RegionPlotContext
from src.utils import norm1

mh.style.use("ATLAS")


class StreamingRingProfileAccumulator:
    """Accumulates normalized ring energy sums across regions using O(1) memory."""

    def __init__(self, n_rings: int, ring_indexes: Optional[np.ndarray] = None) -> None:
        """Initializes running sums for signal and background ring profiles.

        Args:
            n_rings: Number of active calorimeter rings.
            ring_indexes: Optional array mapping column positions to global ring indices.
        """
        self.n_rings: int = int(n_rings)
        self.ring_indexes: Optional[np.ndarray] = ring_indexes
        self.n_signal: int = 0
        self.n_bg: int = 0
        self._sig_sum: np.ndarray = np.zeros(self.n_rings, dtype=np.float64)
        self._sig_sum_sq: np.ndarray = np.zeros(self.n_rings, dtype=np.float64)
        self._bg_sum: np.ndarray = np.zeros(self.n_rings, dtype=np.float64)
        self._bg_sum_sq: np.ndarray = np.zeros(self.n_rings, dtype=np.float64)

    def update(self, rings_data: np.ndarray, labels: np.ndarray) -> None:
        """Normalizes a batch of ring events and updates signal/background running sums.

        Args:
            rings_data: Matrix of shape `(n_events, n_rings)`.
            labels: Binary label vector (`1` for signal, `0` for background).
        """
        normalized = np.asarray(norm1(rings_data), dtype=np.float64)
        sig_mask = labels == 1
        bg_mask = labels == 0

        if np.any(sig_mask):
            sig_vals = normalized[sig_mask]
            self.n_signal += int(sig_vals.shape[0])
            self._sig_sum += np.sum(sig_vals, axis=0)
            self._sig_sum_sq += np.sum(sig_vals * sig_vals, axis=0)

        if np.any(bg_mask):
            bg_vals = normalized[bg_mask]
            self.n_bg += int(bg_vals.shape[0])
            self._bg_sum += np.sum(bg_vals, axis=0)
            self._bg_sum_sq += np.sum(bg_vals * bg_vals, axis=0)

    def compute_stats(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Computes mean and standard deviation per ring for signal and background.

        Returns:
            Tuple `(mean_signal, std_signal, mean_bkg, std_bkg)`.
        """
        if self.n_signal > 0:
            mean_signal = self._sig_sum / self.n_signal
            var_signal = np.maximum(
                self._sig_sum_sq / self.n_signal - mean_signal * mean_signal,
                0.0,
            )
            std_signal = np.sqrt(var_signal)
        else:
            mean_signal = np.zeros(self.n_rings, dtype=np.float64)
            std_signal = np.zeros(self.n_rings, dtype=np.float64)

        if self.n_bg > 0:
            mean_bkg = self._bg_sum / self.n_bg
            var_bkg = np.maximum(
                self._bg_sum_sq / self.n_bg - mean_bkg * mean_bkg,
                0.0,
            )
            std_bkg = np.sqrt(var_bkg)
        else:
            mean_bkg = np.zeros(self.n_rings, dtype=np.float64)
            std_bkg = np.zeros(self.n_rings, dtype=np.float64)

        return mean_signal, std_signal, mean_bkg, std_bkg


class ProfileMeanEnergyPlotter(BasePlotter):
    """Generates average normalized energy profile curves across calorimeter subdetector rings."""

    def __init__(self, language: str = "en") -> None:
        """Initializes the ProfileMeanEnergyPlotter.

        Args:
            language: Target language code ('en' or 'pt').
        """
        super().__init__(name="ProfileMeanEnergyPlotter", language=language)
        mh.style.use("ATLAS")

    def plot_from_stats(
        self,
        mean_signal: np.ndarray,
        std_signal: np.ndarray,
        mean_bkg: np.ndarray,
        std_bkg: np.ndarray,
        output_dir: Union[str, Path],
        filename: str,
        ring_indexes: Optional[np.ndarray] = None,
        plot_title: Optional[str] = None,
        language: Optional[str] = None,
        show: bool = False,
        subfolder: str = "RingsMeanProfiles",
    ) -> str:
        """Renders and saves an energy profile plot directly from precomputed statistics.

        Args:
            mean_signal: Mean normalized energy per ring for signal events.
            std_signal: Standard deviation per ring for signal events.
            mean_bkg: Mean normalized energy per ring for background events.
            std_bkg: Standard deviation per ring for background events.
            output_dir: Base output directory where figures are saved.
            filename: Output PDF filename.
            ring_indexes: Optional array mapping column positions to global ring indices.
            plot_title: Optional custom chart title text.
            language: Optional language override ('en' or 'pt').
            show: Whether to display the figure inline before closing it.
            subfolder: Output subdirectory inside `output_dir`.

        Returns:
            Absolute file path of the generated PDF figure.
        """
        mh.style.use("ATLAS")
        lang = self._resolve_language(language=language)
        resolved_title = (
            plot_title
            if plot_title is not None
            else self.tr(PlotTextKey.PROFILE_TITLE, language=lang)
        )

        n_rings = len(mean_signal)
        x_axis = np.arange(n_rings)
        y_max = max(float(np.max(mean_bkg)), float(np.max(mean_signal)))
        upper_ylim = y_max * 1.5 if y_max > 0.0 else 1.0

        def get_ring_pos(val: int) -> int:
            if ring_indexes is None:
                return 0
            idx = np.where(np.atleast_1d(ring_indexes) == val)[0]
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

        fig, ax = plt.subplots(figsize=(10, 5))

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
            label=self.tr(PlotTextKey.PROFILE_SIGNAL_LABEL, language=lang),
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
            label=self.tr(PlotTextKey.PROFILE_BACKGROUND_LABEL, language=lang),
        )

        text_y = y_max * 1.2 if y_max > 0.0 else 0.75
        for x_pos, name, color in zip(subdet_x, subdet_names, subdet_colors):
            ax.axvline(x=x_pos, color=color, linestyle="--", linewidth=1)
            ax.text(
                x_pos + 1.2,
                text_y,
                name,
                rotation=90,
                va="bottom",
                ha="center",
                fontsize=9,
                color=color,
            )

        ax.set_xlabel(self.tr(PlotTextKey.PROFILE_XLABEL, language=lang), fontsize=13)
        ax.set_ylabel(self.tr(PlotTextKey.PROFILE_YLABEL, language=lang), fontsize=13)

        ticks_step = 10 if n_rings >= 50 else 5
        ticks_loc = np.arange(0, n_rings + 1, ticks_step)
        ax.set_xticks(ticks_loc)
        ax.set_xticklabels([str(i) for i in ticks_loc], fontsize=11)
        ax.tick_params(axis="y", labelsize=11)

        ax.set_xlim(-1, n_rings + 1)
        ax.set_ylim(-0.02, upper_ylim)
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend(fontsize=10, loc="upper right")
        ax.set_title(resolved_title, fontsize=14)
        fig.tight_layout()

        return self.save_figure(fig, output_dir, subfolder, filename, show=show)

    def plot(
        self,
        context: RegionPlotContext,
        x_rings: Optional[np.ndarray] = None,
        plot_name: str = "RingsMeanProfiles_NeuralRinger",
        plot_title: Optional[str] = None,
        show: bool = False,
        **kwargs: Any,
    ) -> str:
        """Renders and saves the mean energy profile comparing signal vs background.

        Args:
            context: RegionPlotContext with kinematic region data.
            x_rings: Optional explicit ring array (defaults to context.x_holdout_rings).
            plot_name: Suffix string for output PDF naming.
            plot_title: Optional custom chart title text.
            show: Whether to display the figure inline before closing it.
            **kwargs: Extra plotting parameters.

        Returns:
            Absolute file path of the generated PDF figure.
        """
        lang = self._resolve_language(context)
        rings_data = x_rings if x_rings is not None else context.x_holdout_rings
        y_holdout = context.y_holdout

        normalized_rings = norm1(rings_data)
        signal_data = normalized_rings[y_holdout == 1]
        bg_data = normalized_rings[y_holdout == 0]

        mean_signal = np.mean(signal_data, axis=0)
        std_signal = np.std(signal_data, axis=0)
        mean_bkg = np.mean(bg_data, axis=0)
        std_bkg = np.std(bg_data, axis=0)

        filename = f"et{context.iet}.eta{context.ieta}.{plot_name}.pdf"
        return self.plot_from_stats(
            mean_signal=mean_signal,
            std_signal=std_signal,
            mean_bkg=mean_bkg,
            std_bkg=std_bkg,
            output_dir=context.output_dir,
            filename=filename,
            ring_indexes=context.ring_indexes,
            plot_title=plot_title,
            language=lang,
            show=show,
        )
