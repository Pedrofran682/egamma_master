import pathlib
from typing import Any, Dict, List, Optional
import matplotlib.pyplot as plt
import numpy as np

import mplhep as mh
mh.style.use("ATLAS")

from src.core.Interfaces.BasePlotter import BasePlotter


class FeatureDistributionPlotter(BasePlotter):
    """Generates two-panel feature distribution and selection ratio plots."""

    def __init__(self, name: str = "FeatureDistributionPlotter") -> None:
        """Initializes the plotter with ATLAS style."""
        super().__init__(name=name)
        mh.style.use("ATLAS")

    def plot_single_feature(
        self,
        feature_data: Dict[str, Any],
        feature_name: str,
        region_key: str = "",
        is_signal: bool = True,
        log_y: bool = False,
        output_path: Optional[pathlib.Path] = None,
        plot_format: str = "pdf",
    ) -> plt.Figure:
        """Renders an individual two-panel distribution and ratio figure.

        Args:
            feature_data: Dictionary containing bin_centers, true_counts, display_title, and models.
            feature_name: Identifier name of feature.
            region_key: Region descriptor.
            is_signal: True for signal distribution, False for background.
            log_y: Whether to use logarithmic scale for upper panel.
            output_path: Destination path for saving figure.
            plot_format: Graphic format extension.

        Returns:
            Matplotlib Figure instance.
        """
        fig, (ax_main, ax_res) = plt.subplots(
            2,
            1,
            figsize=(8, 6.5),
            sharex=True,
            gridspec_kw={"height_ratios": [3.2, 1.0], "hspace": 0.08},
        )

        bin_centers = feature_data["bin_centers"]
        true_counts = feature_data["true_counts"]
        true_err = np.sqrt(true_counts)
        display_title = feature_data.get("display_title", feature_name)
        class_label = "True Signal" if is_signal else "True Background"

        ax_main.plot(
            bin_centers,
            true_counts,
            "o",
            color="black",
            markersize=4,
            zorder=5,
            label=class_label,
        )

        for m in feature_data["models"]:
            counts = m["counts"]
            errors = m["errors"]
            color = m.get("color", "tab:blue")
            marker = m.get("marker", "o")
            is_ml = m.get("is_ml", True)
            label = m.get("label", m.get("name", ""))

            if is_ml:
                ax_main.errorbar(
                    bin_centers,
                    counts,
                    yerr=errors,
                    fmt=marker,
                    color=color,
                    markerfacecolor=color,
                    markeredgecolor=color,
                    linestyle="-",
                    linewidth=1.5,
                    markersize=5,
                    capsize=2,
                    label=label,
                )
            else:
                ax_main.errorbar(
                    bin_centers,
                    counts,
                    yerr=errors,
                    fmt=marker,
                    color=color,
                    markerfacecolor="none",
                    markeredgecolor=color,
                    markeredgewidth=1.8,
                    linestyle="--",
                    linewidth=1.5,
                    markersize=6,
                    capsize=2,
                    label=label,
                )

            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = np.where(true_counts > 0, counts / true_counts, np.nan)
                rel_m = np.where(counts > 0, errors / counts, 0.0)
                rel_true = np.where(true_counts > 0, true_err / true_counts, 0.0)
                ratio_err = ratio * np.sqrt(rel_m**2 + rel_true**2)

            if is_ml:
                ax_res.errorbar(
                    bin_centers,
                    ratio,
                    yerr=ratio_err,
                    fmt=marker,
                    color=color,
                    markerfacecolor=color,
                    markeredgecolor=color,
                    linestyle="-",
                    linewidth=1.2,
                    markersize=4,
                    capsize=2,
                )
            else:
                ax_res.errorbar(
                    bin_centers,
                    ratio,
                    yerr=ratio_err,
                    fmt=marker,
                    color=color,
                    markerfacecolor="none",
                    markeredgecolor=color,
                    markeredgewidth=1.5,
                    linestyle="--",
                    linewidth=1.2,
                    markersize=5,
                    capsize=2,
                )

        ax_res.axhline(1.0, color="gray", linestyle="--", linewidth=1.2, alpha=0.8)
        ax_res.set_ylabel(r"$N_{\mathrm{sel}} / N_{\mathrm{true}}$", fontsize=9)
        ax_res.set_xlabel(display_title, loc="center")
        ax_res.grid(True, linestyle=":", alpha=0.5)

        if is_signal:
            ax_res.set_ylim(0.0, 1.25)
        else:
            ax_res.set_ylim(bottom=0.0)

        prefix = "Signal" if is_signal else "Background"
        title_suffix = f" ({region_key})" if region_key else ""
        ax_main.set_title(f"{prefix}: {display_title}{title_suffix}", pad=12)
        ax_main.set_ylabel(f"{prefix} Events / Bin")

        if log_y:
            ax_main.set_yscale("log")
            ax_main.set_ylim(bottom=0.5)

        ax_main.legend(frameon=True, fontsize=9)
        fig.align_ylabels([ax_main, ax_res])

        if output_path is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_path, format=plot_format, bbox_inches="tight")

        return fig

    def plot_overview_grid(
        self,
        plot_data: Dict[str, Any],
        features_to_plot: List[str],
        region_key: str = "",
        is_signal: bool = True,
        log_y: bool = False,
        cols: int = 2,
    ) -> plt.Figure:
        """Renders an aggregated grid with two-panel distribution subplots.

        Args:
            plot_data: Dictionary mapping feature names to feature plot dictionaries.
            features_to_plot: List of feature keys to display.
            region_key: Region descriptor.
            is_signal: True for signal, False for background.
            log_y: Whether to use logarithmic scale on upper panels.
            cols: Number of column pairs in grid layout.

        Returns:
            Matplotlib Figure instance.
        """
        n_features = len(features_to_plot)
        rows = (n_features + cols - 1) // cols

        fig = plt.figure(figsize=(16, 5.5 * rows))
        gs = fig.add_gridspec(
            rows * 2,
            cols,
            height_ratios=[3.2, 1.0] * rows,
            hspace=0.12,
            wspace=0.25,
        )

        for idx, feat_name in enumerate(features_to_plot):
            r = (idx // cols) * 2
            c = idx % cols
            ax_main = fig.add_subplot(gs[r, c])
            ax_res = fig.add_subplot(gs[r + 1, c], sharex=ax_main)

            s_data = plot_data[feat_name]
            bin_centers = s_data["bin_centers"]
            true_counts = s_data["true_counts"]
            true_err = np.sqrt(true_counts)
            display_title = s_data.get("display_title", feat_name)
            class_label = "True Signal" if is_signal else "True Background"

            ax_main.plot(
                bin_centers,
                true_counts,
                "o",
                color="black",
                markersize=4,
                zorder=5,
                label=class_label,
            )

            for m in s_data["models"]:
                counts = m["counts"]
                errors = m["errors"]
                color = m.get("color", "tab:blue")
                marker = m.get("marker", "o")
                is_ml = m.get("is_ml", True)
                label = m.get("label", m.get("name", ""))

                if is_ml:
                    ax_main.errorbar(
                        bin_centers,
                        counts,
                        yerr=errors,
                        fmt=marker,
                        color=color,
                        markerfacecolor=color,
                        markeredgecolor=color,
                        linestyle="-",
                        linewidth=1.5,
                        markersize=5,
                        capsize=2,
                        label=label,
                    )
                else:
                    ax_main.errorbar(
                        bin_centers,
                        counts,
                        yerr=errors,
                        fmt=marker,
                        color=color,
                        markerfacecolor="none",
                        markeredgecolor=color,
                        markeredgewidth=1.8,
                        linestyle="--",
                        linewidth=1.5,
                        markersize=6,
                        capsize=2,
                        label=label,
                    )

                with np.errstate(divide="ignore", invalid="ignore"):
                    ratio = np.where(true_counts > 0, counts / true_counts, np.nan)
                    rel_m = np.where(counts > 0, errors / counts, 0.0)
                    rel_true = np.where(true_counts > 0, true_err / true_counts, 0.0)
                    ratio_err = ratio * np.sqrt(rel_m**2 + rel_true**2)

                if is_ml:
                    ax_res.errorbar(
                        bin_centers,
                        ratio,
                        yerr=ratio_err,
                        fmt=marker,
                        color=color,
                        markerfacecolor=color,
                        markeredgecolor=color,
                        linestyle="-",
                        linewidth=1.2,
                        markersize=4,
                        capsize=2,
                    )
                else:
                    ax_res.errorbar(
                        bin_centers,
                        ratio,
                        yerr=ratio_err,
                        fmt=marker,
                        color=color,
                        markerfacecolor="none",
                        markeredgecolor=color,
                        markeredgewidth=1.5,
                        linestyle="--",
                        linewidth=1.2,
                        markersize=5,
                        capsize=2,
                    )

            ax_res.axhline(1.0, color="gray", linestyle="--", linewidth=1.2, alpha=0.8)
            ax_res.set_ylabel(r"$N_{\mathrm{sel}} / N_{\mathrm{true}}$", fontsize=8)
            ax_res.grid(True, linestyle=":", alpha=0.5)

            if is_signal:
                ax_res.set_ylim(0.0, 1.25)
            else:
                ax_res.set_ylim(bottom=0.0)

            prefix = "Signal" if is_signal else "Background"
            title_suffix = f" ({region_key})" if region_key else ""
            ax_main.set_title(f"{prefix}: {display_title}{title_suffix}", pad=12)
            ax_main.set_ylabel(f"{prefix} Events / Bin")
            plt.setp(ax_main.get_xticklabels(), visible=False)
            ax_res.set_xlabel(display_title, loc="center")

            if log_y:
                ax_main.set_yscale("log")
                ax_main.set_ylim(bottom=0.5)

            ax_main.legend(frameon=True, fontsize=8)

        total_grid_cells = rows * cols
        for empty_idx in range(n_features, total_grid_cells):
            er = (empty_idx // cols) * 2
            ec = empty_idx % cols
            e_main = fig.add_subplot(gs[er, ec])
            e_res = fig.add_subplot(gs[er + 1, ec])
            fig.delaxes(e_main)
            fig.delaxes(e_res)

        headline = "Signal (Photons)" if is_signal else "Background (Jets)"
        suffix = f" - Region {region_key}" if region_key else ""
        fig.suptitle(f"Holdout {headline} Selection{suffix}", fontsize=14, y=1.002)

        return fig

    def plot_and_export_all(
        self,
        plot_data: Dict[str, Any],
        features_to_plot: List[str],
        output_dir: pathlib.Path,
        region_key: str = "",
        is_signal: bool = True,
        log_y: bool = False,
        plot_format: str = "pdf",
        cols: int = 2,
    ) -> plt.Figure:
        """Exports individual feature figures and returns overview grid.

        Args:
            plot_data: Precomputed distribution plot data.
            features_to_plot: Features to render.
            output_dir: Directory where individual figures are exported.
            region_key: Region identifier.
            is_signal: True for signal, False for background.
            log_y: Whether to use logarithmic scale.
            plot_format: Export graphic format.
            cols: Grid column count.

        Returns:
            Overview grid Matplotlib Figure.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        suffix = "signal_selection" if is_signal else "background_selection"

        for feat_name in features_to_plot:
            single_path = output_dir / f"{feat_name}_{suffix}.{plot_format}"
            fig_single = self.plot_single_feature(
                feature_data=plot_data[feat_name],
                feature_name=feat_name,
                region_key=region_key,
                is_signal=is_signal,
                log_y=log_y,
                output_path=single_path,
                plot_format=plot_format,
            )
            plt.close(fig_single)

        grid_fig = self.plot_overview_grid(
            plot_data=plot_data,
            features_to_plot=features_to_plot,
            region_key=region_key,
            is_signal=is_signal,
            log_y=log_y,
            cols=cols,
        )
        return grid_fig

    def plot(self, context: Any, **kwargs: Any) -> Any:
        """Fulfills BasePlotter abstract method contract.

        Args:
            context: Context object.
            **kwargs: Extra parameters.
        """
        return None

