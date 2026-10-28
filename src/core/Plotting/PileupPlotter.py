import pathlib
from typing import Any, Dict, List, Optional
import matplotlib.pyplot as plt
import numpy as np

import mplhep as mh
mh.style.use("ATLAS")

from src.core.Interfaces.BasePlotter import BasePlotter


class PileupPlotter(BasePlotter):
    """Visualizer for pileup sensitivity, detector drift, and vertex correlation."""

    def __init__(self, name: str = "PileupPlotter") -> None:
        """Initializes PileupPlotter with ATLAS graphic style."""
        super().__init__(name=name)
        mh.style.use("ATLAS")

    def plot_efficiency_vs_mu(
        self,
        models_data: List[Dict[str, Any]],
        title: str = "",
        region_label: str = "",
        is_signal: bool = True,
        output_path: Optional[pathlib.Path] = None,
        file_format: str = "pdf",
    ) -> plt.Figure:
        """Renders 2-panel efficiency or fake rate vs avgmu with bottom ratio to reference.

        Args:
            models_data: List of dictionaries with keys: label, bin_centers, efficiencies,
                uncertainties, color, marker, is_ml, and optional is_reference.
            title: Main title string.
            region_label: Kinematic region descriptor.
            is_signal: True for detection efficiency (PD), False for fake rate (PF).
            output_path: Destination path for figure export.
            file_format: File format extension.

        Returns:
            Matplotlib Figure instance.
        """
        fig, (ax_main, ax_ratio) = plt.subplots(
            2,
            1,
            figsize=(8, 6.5),
            sharex=True,
            gridspec_kw={"height_ratios": [3.2, 1.0], "hspace": 0.08},
        )

        ref_model = next((m for m in models_data if m.get("is_reference", False)), models_data[0])
        ref_eff = ref_model["efficiencies"]
        ref_err = ref_model["uncertainties"]

        for m in models_data:
            centers = m["bin_centers"]
            eff = m["efficiencies"]
            err = m["uncertainties"]
            color = m.get("color", "black")
            marker = m.get("marker", "o")
            is_ml = m.get("is_ml", True)
            label = m.get("label", "")

            if is_ml:
                ax_main.errorbar(
                    centers,
                    eff,
                    yerr=err,
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
                    centers,
                    eff,
                    yerr=err,
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
                ratio = np.where(ref_eff > 0.0, eff / ref_eff, np.nan)
                rel_err_m = np.where(eff > 0.0, err / eff, 0.0)
                rel_err_ref = np.where(ref_eff > 0.0, ref_err / ref_eff, 0.0)
                ratio_err = ratio * np.sqrt(rel_err_m**2 + rel_err_ref**2)

            if is_ml:
                ax_ratio.errorbar(
                    centers,
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
                ax_ratio.errorbar(
                    centers,
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

        ax_ratio.axhline(1.0, color="gray", linestyle="--", linewidth=1.2, alpha=0.8)

        metric_name = "Signal Efficiency ($P_D$)" if is_signal else "Background Fake Rate ($P_F$)"
        full_title = f"{metric_name} vs. $\\langle\\mu\\rangle$"
        if region_label:
            full_title += f" ({region_label})"
        if title:
            full_title = f"{title} - {full_title}"

        ax_main.set_title(full_title, pad=12)
        ax_main.set_ylabel(metric_name)
        ax_main.grid(True, linestyle=":", alpha=0.5)
        ax_main.legend(frameon=True, fontsize=9, loc="best")

        ref_label = ref_model.get("label", "Ref")
        ax_ratio.set_ylabel(f"Ratio to\n{ref_label}", fontsize=8)
        ax_ratio.set_xlabel(r"Average Interactions Per Crossing $\langle\mu\rangle$", loc="center")
        ax_ratio.grid(True, linestyle=":", alpha=0.5)
        ax_ratio.set_ylim(0.85, 1.15)

        fig.align_ylabels([ax_main, ax_ratio])

        if output_path is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_path, format=file_format, bbox_inches="tight")

        return fig

    def plot_profile_vs_mu(
        self,
        profiles_data: List[Dict[str, Any]],
        observable_label: str,
        title: str = "",
        region_label: str = "",
        output_path: Optional[pathlib.Path] = None,
        file_format: str = "pdf",
    ) -> plt.Figure:
        """Plots profile drift of shower shape or ring energy vs avgmu.

        Args:
            profiles_data: List of profile dictionaries with label, bin_centers,
                means, sems, color, marker.
            observable_label: Label for target observable (e.g., L2 Reta).
            title: Title string.
            region_label: Kinematic region descriptor.
            output_path: Destination path for figure export.
            file_format: File format extension.

        Returns:
            Matplotlib Figure instance.
        """
        fig, ax = plt.subplots(figsize=(8, 5))

        for p in profiles_data:
            centers = p["bin_centers"]
            means = p["means"]
            sems = p["sems"]
            color = p.get("color", "tab:blue")
            marker = p.get("marker", "o")
            label = p.get("label", "")

            ax.errorbar(
                centers,
                means,
                yerr=sems,
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

        full_title = f"{observable_label} vs. $\\langle\\mu\\rangle$"
        if region_label:
            full_title += f" ({region_label})"
        if title:
            full_title = f"{title} - {full_title}"

        ax.set_title(full_title, pad=12)
        ax.set_xlabel(r"Average Interactions Per Crossing $\langle\mu\rangle$", loc="center")
        ax.set_ylabel(f"Mean {observable_label}")
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.legend(frameon=True, fontsize=9, loc="best")
        fig.tight_layout()

        if output_path is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_path, format=file_format, bbox_inches="tight")

        return fig

    def plot_vertex_correlation(
        self,
        correlation_result: Dict[str, Any],
        title: str = "",
        region_label: str = "",
        output_path: Optional[pathlib.Path] = None,
        file_format: str = "pdf",
    ) -> plt.Figure:
        """Plots 2D density heatmap and linear regression between NPV and avgmu.

        Args:
            correlation_result: Result dictionary from PileupAnalyzer.compute_vertex_correlation.
            title: Title string.
            region_label: Kinematic region descriptor.
            output_path: Destination path for figure export.
            file_format: File format extension.

        Returns:
            Matplotlib Figure instance.
        """
        fig, ax = plt.subplots(figsize=(8, 6))

        hist = correlation_result["hist2d_counts"].T
        xedges = correlation_result["hist2d_xedges"]
        yedges = correlation_result["hist2d_yedges"]

        mesh = ax.pcolormesh(
            xedges,
            yedges,
            np.ma.masked_where(hist == 0, hist),
            cmap="Blues",
            shading="auto",
        )
        cbar = fig.colorbar(mesh, ax=ax, pad=0.02)
        cbar.set_label("Events / Bin")

        profile = correlation_result["profile"]
        ax.errorbar(
            profile["bin_centers"],
            profile["means"],
            yerr=profile["sems"],
            fmt="s",
            color="darkred",
            markerfacecolor="red",
            markeredgecolor="darkred",
            linestyle="none",
            markersize=5,
            capsize=2,
            label=r"Mean $N_{\text{PV}} \pm \text{SEM}$",
            zorder=4,
        )

        slope = correlation_result["slope"]
        intercept = correlation_result["intercept"]
        r_sq = correlation_result["r_squared"]
        x_fit = np.linspace(xedges[0], xedges[-1], 100)
        y_fit = slope * x_fit + intercept
        fit_label = f"Fit: $N_{{\\text{{PV}}}} = {slope:.2f}\\langle\\mu\\rangle + {intercept:.2f}$\n($R^2 = {r_sq:.3f}$)"
        ax.plot(x_fit, y_fit, color="black", linestyle="--", linewidth=1.8, label=fit_label, zorder=5)

        full_title = r"Primary Vertices $N_{\text{PV}}$ vs. $\langle\mu\rangle$"
        if region_label:
            full_title += f" ({region_label})"
        if title:
            full_title = f"{title} - {full_title}"

        ax.set_title(full_title, pad=12)
        ax.set_xlabel(r"Average Interactions Per Crossing $\langle\mu\rangle$", loc="center")
        ax.set_ylabel(r"Number of Good Primary Vertices ($N_{\text{PV}}$)")
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.legend(frameon=True, fontsize=9, loc="upper left")
        fig.tight_layout()

        if output_path is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_path, format=file_format, bbox_inches="tight")

        return fig

    def plot(self, context: Any, **kwargs: Any) -> Any:
        """Executes rendering conforming to BasePlotter interface.

        Args:
            context: Plotting context data.
            **kwargs: Extra parameters.
        """
        return None

