from pathlib import Path
from typing import TYPE_CHECKING, Any, List, Optional, Tuple, Union
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import mplhep as mh
import numpy as np
import pandas as pd

from src.core.Interfaces.BasePlotter import BasePlotter
from src.core.Plotting.Localization import PlotTextKey
from src.core.Plotting.Context import RegionPlotContext
from src.utils import norm1

if TYPE_CHECKING:
    from src.core.Validation.RingPeakEnergyAnalyzer import RingPeakEnergyAnalyzer

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


class RingPeakEnergyPlotter(BasePlotter):
    """Visualizes subdetector top-k peak ring energy profiles, bars, and regional heatmaps."""

    def __init__(self, language: str = "en", top_k: int = 3) -> None:
        """Initializes the RingPeakEnergyPlotter.

        Args:
            language: Target language code ('en' or 'pt').
            top_k: Number of leading rings per subdetector highlighted as peak rings.
        """
        super().__init__(name="RingPeakEnergyPlotter", language=language)
        mh.style.use("ATLAS")
        from src.core.Validation.RingPeakEnergyAnalyzer import RingPeakEnergyAnalyzer

        self.analyzer: RingPeakEnergyAnalyzer = RingPeakEnergyAnalyzer(top_k=top_k)

    @staticmethod
    def _resolve_file_format(filename: str) -> str:
        suffix = Path(filename).suffix.lstrip(".").lower()
        return suffix if suffix in ("pdf", "png", "svg", "jpg", "jpeg") else "pdf"

    def plot_highlighted_peak_profile(
        self,
        mean_signal: np.ndarray,
        std_signal: np.ndarray,
        mean_bkg: np.ndarray,
        std_bkg: np.ndarray,
        output_dir: Union[str, Path],
        filename: str,
        ring_indexes: Optional[np.ndarray] = None,
        scope_label: str = "Global",
        plot_title: Optional[str] = None,
        language: Optional[str] = None,
        show: bool = False,
        subfolder: str = "RingPeakAnalysis",
    ) -> str:
        """Renders the mean ring energy profile highlighting the top-k peak rings per subdetector.

        Args:
            mean_signal: Mean normalized energy per ring for signal events.
            std_signal: Standard deviation per ring for signal events.
            mean_bkg: Mean normalized energy per ring for background events.
            std_bkg: Standard deviation per ring for background events.
            output_dir: Base directory where the figure will be saved.
            filename: Output filename including extension.
            ring_indexes: Optional array mapping local positions to global ring indices.
            scope_label: Scope descriptor shown in the title (e.g. region name or 'Global').
            plot_title: Optional custom chart title.
            language: Optional language override ('en' or 'pt').
            show: Whether to display the figure inline before closing.
            subfolder: Output subdirectory inside `output_dir`.

        Returns:
            Absolute file path of the saved figure.
        """
        mh.style.use("ATLAS")
        lang = self._resolve_language(language=language)
        n_rings = len(mean_signal)
        resolved_title = (
            plot_title
            if plot_title is not None
            else self.tr(
                PlotTextKey.PEAK_PROFILE_TITLE,
                language=lang,
                top_k=self.analyzer.top_k,
                n_rings=n_rings,
                scope=scope_label,
            )
        )

        comparison_df = self.analyzer.analyze_signal_vs_background(
            mean_signal=mean_signal,
            mean_bkg=mean_bkg,
            ring_indexes=ring_indexes,
            region=scope_label,
        )
        positions_map = self.analyzer.resolve_subdetector_positions(
            n_rings=n_rings,
            ring_indexes=ring_indexes,
        )

        x_axis = np.arange(n_rings)
        y_max = max(float(np.max(mean_bkg)), float(np.max(mean_signal)))
        upper_ylim = y_max * 1.9 if y_max > 0.0 else 1.0

        fig, ax = plt.subplots(figsize=(12, 6))

        highlight_label = self.tr(
            PlotTextKey.PEAK_HIGHLIGHT_LABEL,
            language=lang,
            top_k=self.analyzer.top_k,
        )
        added_span_legend = False
        text_y = y_max * 1.08 if y_max > 0.0 else 0.55

        for subdet in self.analyzer.subdetectors:
            pos = positions_map[subdet.name]
            if pos.size == 0:
                continue
            n_peak = min(self.analyzer.top_k, int(pos.size))
            peak_pos = pos[:n_peak]
            span_start = float(peak_pos[0]) - 0.4
            span_end = float(peak_pos[-1]) + 0.4

            ax.axvspan(
                span_start,
                span_end,
                color=subdet.color,
                alpha=0.18,
                label=highlight_label if not added_span_legend else None,
            )
            added_span_legend = True
            ax.axvline(x=float(pos[0]), color=subdet.color, linestyle="--", linewidth=1.1)

            row_match = comparison_df[comparison_df["subdetector"] == subdet.name]
            if not row_match.empty:
                sig_pct = float(row_match["signal_intra_detector_pct"].iloc[0])
                bkg_pct = float(row_match["background_intra_detector_pct"].iloc[0])
                badge = f"{subdet.name} ({sig_pct:.1f}% / {bkg_pct:.1f}%)"
            else:
                badge = subdet.name

            ax.text(
                float(pos[0]) + 0.5,
                text_y,
                badge,
                rotation=90,
                va="bottom",
                ha="left",
                fontsize=8.5,
                color=subdet.color,
                bbox=dict(
                    boxstyle="round,pad=0.15",
                    facecolor="white",
                    edgecolor=subdet.color,
                    alpha=0.85,
                ),
            )

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
            zorder=4,
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
            zorder=4,
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
        ax.grid(True, linestyle="--", alpha=0.55)
        ax.legend(fontsize=9.5, loc="upper center", ncol=3, frameon=True)
        ax.set_title(resolved_title, fontsize=13.5, pad=10)
        fig.tight_layout()

        return self.save_figure(
            fig=fig,
            folder_path=output_dir,
            subfolder=subfolder,
            filename=filename,
            file_format=self._resolve_file_format(filename),
            show=show,
        )

    def plot_subdetector_peak_bars(
        self,
        comparison_df: pd.DataFrame,
        output_dir: Union[str, Path],
        filename: str,
        scope_label: str = "Global",
        plot_title: Optional[str] = None,
        language: Optional[str] = None,
        show: bool = False,
        subfolder: str = "RingPeakAnalysis",
    ) -> str:
        """Renders grouped bar charts of intra-detector and global peak energy percentages.

        Args:
            comparison_df: Output DataFrame from `RingPeakEnergyAnalyzer.analyze_signal_vs_background`.
            output_dir: Base directory where the figure will be saved.
            filename: Output filename including extension.
            scope_label: Scope descriptor shown in the figure suptitle.
            plot_title: Optional custom suptitle string.
            language: Optional language override ('en' or 'pt').
            show: Whether to display the figure inline before closing.
            subfolder: Output subdirectory inside `output_dir`.

        Returns:
            Absolute file path of the saved figure.
        """
        mh.style.use("ATLAS")
        lang = self._resolve_language(language=language)

        subdets = comparison_df["subdetector"].tolist()
        x = np.arange(len(subdets))
        width = 0.36

        sig_intra = comparison_df["signal_intra_detector_pct"].to_numpy(dtype=float)
        bkg_intra = comparison_df["background_intra_detector_pct"].to_numpy(dtype=float)

        sig_global = comparison_df["signal_global_calorimeter_pct"].to_numpy(dtype=float)
        bkg_global = comparison_df["background_global_calorimeter_pct"].to_numpy(dtype=float)
        sig_share = comparison_df["signal_subdetector_share_pct"].to_numpy(dtype=float)
        bkg_share = comparison_df["background_subdetector_share_pct"].to_numpy(dtype=float)

        n_rings_model = (
            int(comparison_df["n_rings_model"].iloc[0])
            if "n_rings_model" in comparison_df.columns and not comparison_df.empty
            else 100
        )

        sig_label = self.tr(PlotTextKey.PROFILE_SIGNAL_LABEL, language=lang)
        bkg_label = self.tr(PlotTextKey.PROFILE_BACKGROUND_LABEL, language=lang)
        peak_label = self.tr(
            PlotTextKey.PEAK_HIGHLIGHT_LABEL,
            language=lang,
            top_k=self.analyzer.top_k,
        )
        full_subdet_label = self.tr(PlotTextKey.PEAK_TOTAL_SUBDET_LABEL, language=lang)

        fig, (ax_intra, ax_global) = plt.subplots(1, 2, figsize=(15, 6))

        bars_sig_intra = ax_intra.bar(
            x - width / 2,
            sig_intra,
            width,
            color="navy",
            alpha=0.9,
            label=sig_label,
        )
        bars_bkg_intra = ax_intra.bar(
            x + width / 2,
            bkg_intra,
            width,
            color="darkorange",
            alpha=0.9,
            label=bkg_label,
        )

        for bar in list(bars_sig_intra) + list(bars_bkg_intra):
            height = bar.get_height()
            ax_intra.annotate(
                f"{height:.1f}%",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8.5,
            )

        ax_intra.set_xticks(x)
        ax_intra.set_xticklabels(subdets, fontsize=10)
        ax_intra.set_xlabel(self.tr(PlotTextKey.PEAK_SUBDETECTOR_XLABEL, language=lang), fontsize=11)
        ax_intra.set_ylabel(self.tr(PlotTextKey.PEAK_INTRA_YLABEL, language=lang), fontsize=11)
        ax_intra.set_title(
            self.tr(PlotTextKey.PEAK_INTRA_BAR_TITLE, language=lang, top_k=self.analyzer.top_k),
            fontsize=11.5,
            pad=8,
        )
        max_intra = max(float(np.max(sig_intra)), float(np.max(bkg_intra)), 1.0)
        ax_intra.set_ylim(0.0, max(108.0, max_intra * 1.15))
        ax_intra.grid(True, axis="y", linestyle="--", alpha=0.5)
        ax_intra.legend(fontsize=9.5, loc="upper right", frameon=True)

        ax_global.bar(
            x - width / 2,
            sig_share,
            width,
            color="navy",
            alpha=0.22,
            edgecolor="navy",
            hatch="//",
            linewidth=1.0,
        )
        ax_global.bar(
            x + width / 2,
            bkg_share,
            width,
            color="darkorange",
            alpha=0.22,
            edgecolor="darkorange",
            hatch="//",
            linewidth=1.0,
        )
        bars_sig_global = ax_global.bar(
            x - width / 2,
            sig_global,
            width,
            color="navy",
            alpha=0.92,
        )
        bars_bkg_global = ax_global.bar(
            x + width / 2,
            bkg_global,
            width,
            color="darkorange",
            alpha=0.92,
        )

        for idx, (b_sig, b_bkg) in enumerate(zip(bars_sig_global, bars_bkg_global)):
            h_sig = b_sig.get_height()
            h_bkg = b_bkg.get_height()
            ax_global.annotate(
                f"{h_sig:.1f}%",
                xy=(b_sig.get_x() + b_sig.get_width() / 2, max(h_sig, sig_share[idx])),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8.5,
                color="navy",
                fontweight="bold",
            )
            ax_global.annotate(
                f"{h_bkg:.1f}%",
                xy=(b_bkg.get_x() + b_bkg.get_width() / 2, max(h_bkg, bkg_share[idx])),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8.5,
                color="darkorange",
                fontweight="bold",
            )

        global_legend_handles = [
            Patch(facecolor="navy", alpha=0.92, label=f"{sig_label} ({peak_label})"),
            Patch(facecolor="darkorange", alpha=0.92, label=f"{bkg_label} ({peak_label})"),
            Patch(
                facecolor="gray",
                alpha=0.22,
                edgecolor="black",
                hatch="//",
                label=full_subdet_label,
            ),
        ]
        ax_global.set_xticks(x)
        ax_global.set_xticklabels(subdets, fontsize=10)
        ax_global.set_xlabel(
            self.tr(PlotTextKey.PEAK_SUBDETECTOR_XLABEL, language=lang),
            fontsize=11,
        )
        ax_global.set_ylabel(self.tr(PlotTextKey.PEAK_GLOBAL_YLABEL, language=lang), fontsize=11)
        ax_global.set_title(
            self.tr(PlotTextKey.PEAK_GLOBAL_BAR_TITLE, language=lang, top_k=self.analyzer.top_k),
            fontsize=11.5,
            pad=8,
        )
        max_global = max(
            float(np.max(sig_share)),
            float(np.max(bkg_share)),
            float(np.max(sig_global)),
            float(np.max(bkg_global)),
            1.0,
        )
        ax_global.set_ylim(0.0, max_global * 1.22)
        ax_global.grid(True, axis="y", linestyle="--", alpha=0.5)
        ax_global.legend(handles=global_legend_handles, fontsize=8.5, loc="upper right", frameon=True)

        suptitle = (
            plot_title
            if plot_title is not None
            else self.tr(
                PlotTextKey.PEAK_BAR_SUPTITLE,
                language=lang,
                n_rings=n_rings_model,
                scope=scope_label,
            )
        )
        fig.suptitle(suptitle, fontsize=13.5, y=0.98)
        fig.tight_layout(rect=[0, 0, 1, 0.92])

        return self.save_figure(
            fig=fig,
            folder_path=output_dir,
            subfolder=subfolder,
            filename=filename,
            file_format=self._resolve_file_format(filename),
            show=show,
        )

    def _render_variance_comparison_panel(
        self,
        ax: plt.Axes,
        x: np.ndarray,
        width: float,
        subdets: List[str],
        sig_all: np.ndarray,
        sig_peak: np.ndarray,
        bkg_all: np.ndarray,
        bkg_peak: np.ndarray,
        ylabel: str,
        panel_title: str,
        value_fmt: str,
        use_log: bool,
        legend_handles: List[Patch],
        lang: str,
    ) -> None:
        ax.bar(
            x - width / 2,
            sig_all,
            width,
            color="navy",
            alpha=0.28,
            edgecolor="navy",
            hatch="//",
            linewidth=1.0,
            zorder=2,
        )
        ax.bar(
            x + width / 2,
            bkg_all,
            width,
            color="darkorange",
            alpha=0.28,
            edgecolor="darkorange",
            hatch="//",
            linewidth=1.0,
            zorder=2,
        )
        bars_sig = ax.bar(
            x - width / 2,
            sig_peak,
            width * 0.56,
            color="navy",
            alpha=0.90,
            zorder=3,
        )
        bars_bkg = ax.bar(
            x + width / 2,
            bkg_peak,
            width * 0.56,
            color="darkorange",
            alpha=0.90,
            zorder=3,
        )

        for idx, (b_sig, b_bkg) in enumerate(zip(bars_sig, bars_bkg)):
            h_sig = b_sig.get_height()
            h_bkg = b_bkg.get_height()
            ax.annotate(
                f"{h_sig:{value_fmt}}",
                xy=(b_sig.get_x() + b_sig.get_width() / 2, max(h_sig, sig_all[idx])),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8.0,
                color="navy",
                fontweight="bold",
            )
            ax.annotate(
                f"{h_bkg:{value_fmt}}",
                xy=(b_bkg.get_x() + b_bkg.get_width() / 2, max(h_bkg, bkg_all[idx])),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8.0,
                color="darkorange",
                fontweight="bold",
            )

        all_vals = np.concatenate([sig_all, sig_peak, bkg_all, bkg_peak])
        if use_log and all_vals.size > 0 and np.all(all_vals > 0.0):
            ax.set_yscale("log")
            ax.set_ylim(float(np.min(all_vals)) * 0.4, float(np.max(all_vals)) * 5.0)
        else:
            max_val = max(float(np.max(all_vals)) if all_vals.size > 0 else 1.0, 1e-6)
            ax.set_ylim(0.0, max_val * 1.24)

        ax.set_xticks(x)
        ax.set_xticklabels(subdets, fontsize=10)
        ax.set_xlabel(self.tr(PlotTextKey.PEAK_SUBDETECTOR_XLABEL, language=lang), fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(panel_title, fontsize=11.0, pad=8)
        ax.grid(True, axis="y", linestyle="--", alpha=0.5)
        ax.legend(handles=legend_handles, fontsize=8.5, loc="upper right", frameon=True)

    def plot_subdetector_variance_bars(
        self,
        comparison_df: pd.DataFrame,
        output_dir: Union[str, Path],
        filename: str,
        scope_label: str = "Global",
        plot_title: Optional[str] = None,
        language: Optional[str] = None,
        show: bool = False,
        subfolder: str = "RingPeakAnalysis",
    ) -> str:
        """Renders grouped bars comparing mean ring variance and relative CV per subdetector.

        Args:
            comparison_df: Output DataFrame from `RingPeakEnergyAnalyzer.analyze_signal_vs_background`.
            output_dir: Base directory where the figure will be saved.
            filename: Output filename including extension.
            scope_label: Scope descriptor shown in the figure suptitle.
            plot_title: Optional custom suptitle string.
            language: Optional language override ('en' or 'pt').
            show: Whether to display the figure inline before closing.
            subfolder: Output subdirectory inside `output_dir`.

        Returns:
            Absolute file path of the saved figure.
        """
        mh.style.use("ATLAS")
        lang = self._resolve_language(language=language)

        subdets = comparison_df["subdetector"].tolist()
        x = np.arange(len(subdets))
        width = 0.36

        sig_var_all = comparison_df["signal_mean_ring_variance"].to_numpy(dtype=float)
        sig_var_peak = comparison_df["signal_peak_mean_ring_variance"].to_numpy(dtype=float)
        bkg_var_all = comparison_df["background_mean_ring_variance"].to_numpy(dtype=float)
        bkg_var_peak = comparison_df["background_peak_mean_ring_variance"].to_numpy(dtype=float)

        sig_cv_all = comparison_df["signal_subdetector_cv"].to_numpy(dtype=float)
        sig_cv_peak = comparison_df["signal_peak_cv"].to_numpy(dtype=float)
        bkg_cv_all = comparison_df["background_subdetector_cv"].to_numpy(dtype=float)
        bkg_cv_peak = comparison_df["background_peak_cv"].to_numpy(dtype=float)

        n_rings_model = (
            int(comparison_df["n_rings_model"].iloc[0])
            if "n_rings_model" in comparison_df.columns and not comparison_df.empty
            else 100
        )

        sig_label = self.tr(PlotTextKey.PROFILE_SIGNAL_LABEL, language=lang)
        bkg_label = self.tr(PlotTextKey.PROFILE_BACKGROUND_LABEL, language=lang)
        peak_label = self.tr(
            PlotTextKey.PEAK_HIGHLIGHT_LABEL,
            language=lang,
            top_k=self.analyzer.top_k,
        )
        full_subdet_label = self.tr(PlotTextKey.PEAK_TOTAL_SUBDET_LABEL, language=lang)

        legend_handles = [
            Patch(facecolor="navy", alpha=0.90, label=f"{sig_label} ({peak_label})"),
            Patch(facecolor="darkorange", alpha=0.90, label=f"{bkg_label} ({peak_label})"),
            Patch(
                facecolor="gray",
                alpha=0.28,
                edgecolor="black",
                hatch="//",
                label=full_subdet_label,
            ),
        ]

        fig, (ax_var, ax_cv) = plt.subplots(1, 2, figsize=(15, 6))
        self._render_variance_comparison_panel(
            ax=ax_var,
            x=x,
            width=width,
            subdets=subdets,
            sig_all=sig_var_all,
            sig_peak=sig_var_peak,
            bkg_all=bkg_var_all,
            bkg_peak=bkg_var_peak,
            ylabel=self.tr(PlotTextKey.PEAK_VARIANCE_YLABEL, language=lang),
            panel_title=self.tr(
                PlotTextKey.PEAK_VARIANCE_ABS_PANEL_TITLE,
                language=lang,
                top_k=self.analyzer.top_k,
            ),
            value_fmt=".1e",
            use_log=True,
            legend_handles=legend_handles,
            lang=lang,
        )
        self._render_variance_comparison_panel(
            ax=ax_cv,
            x=x,
            width=width,
            subdets=subdets,
            sig_all=sig_cv_all,
            sig_peak=sig_cv_peak,
            bkg_all=bkg_cv_all,
            bkg_peak=bkg_cv_peak,
            ylabel=self.tr(PlotTextKey.PEAK_CV_YLABEL, language=lang),
            panel_title=self.tr(PlotTextKey.PEAK_VARIANCE_CV_PANEL_TITLE, language=lang),
            value_fmt=".2f",
            use_log=False,
            legend_handles=legend_handles,
            lang=lang,
        )

        suptitle = (
            plot_title
            if plot_title is not None
            else self.tr(
                PlotTextKey.PEAK_VARIANCE_BAR_SUPTITLE,
                language=lang,
                n_rings=n_rings_model,
                scope=scope_label,
            )
        )
        fig.suptitle(suptitle, fontsize=13.5, y=0.98)
        fig.tight_layout(rect=[0, 0, 1, 0.92])

        return self.save_figure(
            fig=fig,
            folder_path=output_dir,
            subfolder=subfolder,
            filename=filename,
            file_format=self._resolve_file_format(filename),
            show=show,
        )

    def plot_regional_peak_heatmap(
        self,
        regional_df: pd.DataFrame,
        output_dir: Union[str, Path],
        filename: str,
        metric: str = "intra_detector_pct",
        plot_title: Optional[str] = None,
        language: Optional[str] = None,
        show: bool = False,
        subfolder: str = "RingPeakAnalysis",
    ) -> str:
        """Renders a 2-panel heatmap comparing regional peak percentages for Signal and Background.

        Args:
            regional_df: Concatenated DataFrame of regional `analyze_signal_vs_background` results.
            output_dir: Base directory where the figure will be saved.
            filename: Output filename including extension.
            metric: Metric suffix ('intra_detector_pct' or 'global_calorimeter_pct').
            plot_title: Optional custom title override.
            language: Optional language override ('en' or 'pt').
            show: Whether to display the figure inline before closing.
            subfolder: Output subdirectory inside `output_dir`.

        Returns:
            Absolute file path of the saved figure.

        Raises:
            ValueError: If `metric` is unsupported or `regional_df` is empty.
        """
        supported_metrics = (
            "intra_detector_pct",
            "global_calorimeter_pct",
            "mean_ring_variance",
            "peak_mean_ring_variance",
            "subdetector_cv",
            "peak_cv",
        )
        if metric not in supported_metrics:
            raise ValueError(
                f"Unsupported metric '{metric}'. Expected one of {supported_metrics}."
            )
        if regional_df.empty:
            raise ValueError("regional_df must not be empty.")

        mh.style.use("ATLAS")
        lang = self._resolve_language(language=language)

        ordered_subdets = [
            s.name
            for s in self.analyzer.subdetectors
            if s.name in set(regional_df["subdetector"].unique())
        ]
        sig_pivot = (
            regional_df.pivot(index="region", columns="subdetector", values=f"signal_{metric}")
            .reindex(columns=ordered_subdets)
            .sort_index()
        )
        bkg_pivot = (
            regional_df.pivot(index="region", columns="subdetector", values=f"background_{metric}")
            .reindex(columns=ordered_subdets)
            .sort_index()
        )

        sig_vals = sig_pivot.to_numpy(dtype=float)
        bkg_vals = bkg_pivot.to_numpy(dtype=float)
        combined_vals = np.concatenate([sig_vals.ravel(), bkg_vals.ravel()])
        finite_vals = combined_vals[np.isfinite(combined_vals)]
        vmin = float(np.min(finite_vals)) if finite_vals.size > 0 else 0.0
        vmax = float(np.max(finite_vals)) if finite_vals.size > 0 else 100.0
        if np.isclose(vmin, vmax):
            vmax = vmin + 1.0

        n_regions = len(sig_pivot.index)
        fig_height = max(5.5, 0.36 * n_regions + 2.2)
        fig, axes = plt.subplots(1, 2, figsize=(15, fig_height), sharey=True)

        metric_key_map = {
            "intra_detector_pct": PlotTextKey.PEAK_INTRA_YLABEL,
            "global_calorimeter_pct": PlotTextKey.PEAK_GLOBAL_YLABEL,
            "mean_ring_variance": PlotTextKey.PEAK_VARIANCE_YLABEL,
            "peak_mean_ring_variance": PlotTextKey.PEAK_VARIANCE_YLABEL,
            "subdetector_cv": PlotTextKey.PEAK_CV_YLABEL,
            "peak_cv": PlotTextKey.PEAK_CV_YLABEL,
        }
        metric_label = self.tr(metric_key_map[metric], language=lang)
        class_titles = (
            self.tr(PlotTextKey.PROFILE_SIGNAL_LABEL, language=lang),
            self.tr(PlotTextKey.PROFILE_BACKGROUND_LABEL, language=lang),
        )
        pivots = (sig_pivot, bkg_pivot)

        im = None
        for ax, pivot, cls_label in zip(axes, pivots, class_titles):
            matrix = pivot.to_numpy(dtype=float)
            im = ax.imshow(matrix, cmap="YlGnBu", aspect="auto", vmin=vmin, vmax=vmax)

            ax.set_xticks(np.arange(len(ordered_subdets)))
            ax.set_xticklabels(ordered_subdets, fontsize=10, rotation=20, ha="right")
            ax.set_yticks(np.arange(len(pivot.index)))
            ax.set_yticklabels(list(pivot.index), fontsize=9)
            ax.set_xlabel(self.tr(PlotTextKey.PEAK_REGION_XLABEL, language=lang), fontsize=11)

            panel_title = (
                plot_title
                if plot_title is not None
                else self.tr(
                    PlotTextKey.PEAK_HEATMAP_TITLE,
                    language=lang,
                    top_k=self.analyzer.top_k,
                    metric_label=metric_label,
                    class_label=cls_label,
                )
            )
            ax.set_title(panel_title, fontsize=11, pad=10)

            midpoint = 0.5 * (vmin + vmax)
            for row_idx in range(matrix.shape[0]):
                for col_idx in range(matrix.shape[1]):
                    val = matrix[row_idx, col_idx]
                    if np.isfinite(val):
                        text_color = "white" if val > midpoint else "black"
                        if metric.endswith("_pct"):
                            cell_text = f"{val:.1f}%"
                        elif "variance" in metric:
                            cell_text = f"{val:.1e}"
                        else:
                            cell_text = f"{val:.2f}"
                        ax.text(
                            col_idx,
                            row_idx,
                            cell_text,
                            ha="center",
                            va="center",
                            color=text_color,
                            fontsize=8.0 if "variance" in metric else 8.5,
                        )

        axes[0].set_ylabel(self.tr(PlotTextKey.PEAK_REGION_YLABEL, language=lang), fontsize=11)
        fig.tight_layout(rect=[0, 0, 0.92, 1])
        if im is not None:
            cbar_ax = fig.add_axes([0.935, 0.15, 0.018, 0.72])
            cbar = fig.colorbar(im, cax=cbar_ax)
            cbar.set_label(metric_label, fontsize=10.5)

        return self.save_figure(
            fig=fig,
            folder_path=output_dir,
            subfolder=subfolder,
            filename=filename,
            file_format=self._resolve_file_format(filename),
            show=show,
        )

    def plot(
        self,
        context: RegionPlotContext,
        x_rings: Optional[np.ndarray] = None,
        plot_name: str = "RingPeakEnergyBars",
        plot_title: Optional[str] = None,
        show: bool = False,
        **kwargs: Any,
    ) -> str:
        """Computes peak ring metrics from context and renders the subdetector bar chart.

        Args:
            context: RegionPlotContext containing kinematic region ring data and labels.
            x_rings: Optional explicit ring matrix overriding `context.x_holdout_rings`.
            plot_name: Suffix string for output PDF naming.
            plot_title: Optional custom chart suptitle.
            show: Whether to display the figure inline before closing.
            **kwargs: Extra plotting parameters.

        Returns:
            Absolute file path of the generated PDF figure.
        """
        lang = self._resolve_language(context)
        rings_data = x_rings if x_rings is not None else context.x_holdout_rings
        y_holdout = context.y_holdout

        normalized_rings = norm1(rings_data)
        sig_mask = y_holdout == 1
        bg_mask = y_holdout == 0

        mean_signal = (
            np.mean(normalized_rings[sig_mask], axis=0)
            if np.any(sig_mask)
            else np.zeros(normalized_rings.shape[1], dtype=np.float64)
        )
        std_signal = (
            np.std(normalized_rings[sig_mask], axis=0)
            if np.any(sig_mask)
            else np.zeros(normalized_rings.shape[1], dtype=np.float64)
        )
        mean_bkg = (
            np.mean(normalized_rings[bg_mask], axis=0)
            if np.any(bg_mask)
            else np.zeros(normalized_rings.shape[1], dtype=np.float64)
        )
        std_bkg = (
            np.std(normalized_rings[bg_mask], axis=0)
            if np.any(bg_mask)
            else np.zeros(normalized_rings.shape[1], dtype=np.float64)
        )

        region_label = f"et{context.iet}_eta{context.ieta}"
        comparison_df = self.analyzer.analyze_signal_vs_background(
            mean_signal=mean_signal,
            mean_bkg=mean_bkg,
            ring_indexes=context.ring_indexes,
            region=region_label,
            std_signal=std_signal,
            std_bkg=std_bkg,
        )
        filename = f"et{context.iet}.eta{context.ieta}.{plot_name}.pdf"
        return self.plot_subdetector_peak_bars(
            comparison_df=comparison_df,
            output_dir=context.output_dir,
            filename=filename,
            scope_label=region_label,
            plot_title=plot_title,
            language=lang,
            show=show,
        )
