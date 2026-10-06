import logging
import os
import pathlib
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np

import mplhep as mh
mh.style.use("ATLAS")

from src.core.Interfaces.BasePlotter import BasePlotter
from src.core.Plotting.RegionDistributionPlotter import DEFAULT_ET_INTERVALS, DEFAULT_ETA_INTERVALS
from src.core.Validation.QuadrantAnalyzer import RegionalQuadrantResult

log = logging.getLogger()

VARIABLE_LABELS: Dict[str, str] = {
    "Rcore": r"L2 $R_{\text{core}}$ ($E_{237} / E_{277}$)",
    "Rhad": r"L2 $R_{\text{had}}$ ($E_{\text{had1}} / E_T$)",
    "Eratio": r"L2 $E_{\text{ratio}}$ ($(E_{\text{max1}} - E_{\text{max2}}) / (E_{\text{max1}} + E_{\text{max2}})$)",
    "trig_L2_calo_weta2": r"L2 $w_{\eta 2}$ (Lateral width in EM2)",
    "trig_L2_calo_wstot": r"L2 $w_{\text{stot}}$ (Total width in strips)",
    "trig_L2_calo_fracs1": r"L2 $f_{1}$ (Energy fraction in EM1)",
    "trig_L2_calo_ehad1": r"L2 $E_{\text{had1}}$ [MeV]",
    "trig_L2_calo_emaxs1": r"L2 $E_{\text{max1}}$ [MeV]",
    "trig_L2_calo_e2tsts1": r"L2 $E_{\text{max2}}$ [MeV]",
    "trig_L2_calo_e237": r"L2 $E_{237}$ [MeV]",
    "trig_L2_calo_e277": r"L2 $E_{277}$ [MeV]",
}


class QuadrantPlotter(BasePlotter):
    """Generates trigger shower shape quadrant histograms, efficiency curves, and summaries."""

    def __init__(
        self,
        et_intervals: Optional[Dict[int, str]] = None,
        eta_intervals: Optional[Dict[int, str]] = None,
    ) -> None:
        """Initializes QuadrantPlotter with optional kinematic label intervals.

        Args:
            et_intervals: Optional dictionary mapping ET bin indices to physical intervals.
            eta_intervals: Optional dictionary mapping eta bin indices to physical intervals.
        """
        super().__init__(name="QuadrantPlotter")
        self.et_intervals: Dict[int, str] = et_intervals or DEFAULT_ET_INTERVALS
        self.eta_intervals: Dict[int, str] = eta_intervals or DEFAULT_ETA_INTERVALS

    def _get_region_label(self, iet: int, ieta: int) -> str:
        if iet < 0 or ieta < 0:
            return "Global (All Regions)"
        et_desc = self.et_intervals.get(iet, f"Bin {iet}")
        eta_desc = self.eta_intervals.get(ieta, f"Bin {ieta}")
        return f"$E_T$ {et_desc}, $\\eta$ {eta_desc} (iet{iet}.ieta{ieta})"

    def plot_showershape_histogram(
        self,
        result: RegionalQuadrantResult,
        var_name: str,
        output_dir: str | pathlib.Path,
        file_format: str = "pdf",
        nbins: int = 40,
    ) -> Optional[str]:
        """Renders 1D histogram of a shower shape variable partitioned by the 4 quadrant categories.

        Args:
            result: RegionalQuadrantResult containing predictions, labels, and shower shapes.
            var_name: Key name of the shower shape variable to plot.
            output_dir: Destination directory.
            file_format: Graphic format ('pdf' or 'png').
            nbins: Number of histogram bins.

        Returns:
            Saved figure path or None if variable is missing.
        """
        if var_name not in result.showershapes:
            return None

        values = result.showershapes[var_name]
        labels = result.labels
        preds1 = result.preds_strategy1
        preds2 = result.preds_strategy2

        c1 = preds1 == labels
        c2 = preds2 == labels

        categories = np.empty(len(labels), dtype=object)
        categories[c1 & c2] = "Both Correct"
        categories[c1 & ~c2] = f"{result.strategy1_name} Only"
        categories[~c1 & c2] = f"{result.strategy2_name} Only"
        categories[~c1 & ~c2] = "Both Wrong"

        palette = {
            "Both Correct": "#2ca02c",
            f"{result.strategy1_name} Only": "#1f77b4",
            f"{result.strategy2_name} Only": "#ff7f0e",
            "Both Wrong": "#d62728",
        }

        finite_mask = np.isfinite(values)
        if not np.any(finite_mask):
            return None

        clean_vals = values[finite_mask]
        q_low, q_high = float(np.percentile(clean_vals, 0.5)), float(np.percentile(clean_vals, 99.5))
        if q_low == q_high:
            q_low -= 1.0
            q_high += 1.0
        bins = np.linspace(q_low, q_high, nbins + 1)

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        region_str = self._get_region_label(result.iet, result.ieta)
        var_display = VARIABLE_LABELS.get(var_name, var_name)

        fig.suptitle(
            f"{var_display} — Quadrant Breakdown\n{region_str} ({result.strategy1_name} vs {result.strategy2_name})",
            fontsize=13,
            weight="bold",
        )

        sample_masks = [
            ("Signal (Photons)", labels == 1, axes[0]),
            ("Background (Fakes)", labels == 0, axes[1]),
        ]

        for sample_title, s_mask, ax in sample_masks:
            ax.set_title(sample_title, fontsize=12, weight="bold")
            for cat_name, color in palette.items():
                cat_mask = s_mask & (categories == cat_name) & finite_mask
                cat_vals = values[cat_mask]
                count = len(cat_vals)
                if count > 0:
                    ax.hist(
                        cat_vals,
                        bins=bins,
                        histtype="step",
                        linewidth=1.8,
                        color=color,
                        label=f"{cat_name} (N={count:,})",
                    )
                    ax.hist(
                        cat_vals,
                        bins=bins,
                        histtype="stepfilled",
                        alpha=0.10,
                        color=color,
                    )
                else:
                    ax.plot([], [], color=color, label=f"{cat_name} (N=0)")

            ax.set_xlabel(var_display, fontsize=11)
            ax.set_ylabel("Events", fontsize=11)
            ax.grid(True, linestyle=":", alpha=0.5)
            ax.legend(loc="upper right", framealpha=0.9, fontsize=9)

        plt.tight_layout()
        prefix = f"iet{result.iet}_ieta{result.ieta}" if result.iet >= 0 else "global"
        filename = f"{prefix}_{var_name}_quadrant_hist.{file_format}"
        return self.save_figure(fig, output_dir, "ShowershapeHistograms", filename, file_format=file_format)

    @staticmethod
    def _compute_binned_efficiency(
        values: np.ndarray,
        preds: np.ndarray,
        target_class: int,
        bins: int = 10,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Computes binned classification efficiency and binomial errors for a given class.

        Args:
            values: Continuous feature values (ET or eta) for events of the target class.
            preds: Binary predictions for events of the target class.
            target_class: Target class label (1 for signal, 0 for background).
            bins: Number of equal-width bins.

        Returns:
            Tuple of (bin_centers, efficiencies, binomial_errors).
        """
        if len(values) == 0:
            return np.array([]), np.array([]), np.array([])

        finite_mask = np.isfinite(values)
        vals = values[finite_mask]
        p = preds[finite_mask]

        if len(vals) == 0:
            return np.array([]), np.array([]), np.array([])

        q_low, q_high = float(np.percentile(vals, 0.5)), float(np.percentile(vals, 99.5))
        if q_low >= q_high:
            q_low -= 1.0
            q_high += 1.0

        bin_edges = np.linspace(q_low, q_high, bins + 1)
        bin_centers: List[float] = []
        efficiencies: List[float] = []
        binomial_errors: List[float] = []

        for i in range(len(bin_edges) - 1):
            low, high = bin_edges[i], bin_edges[i + 1]
            if i == len(bin_edges) - 2:
                in_bin = (vals >= low) & (vals <= high)
            else:
                in_bin = (vals >= low) & (vals < high)

            n_in_bin = int(np.sum(in_bin))
            if n_in_bin > 0:
                n_correct = int(np.sum(p[in_bin] == target_class))
                eff = float(n_correct / n_in_bin)
                err = float(np.sqrt((eff * (1.0 - eff)) / n_in_bin))

                bin_centers.append((low + high) / 2.0)
                efficiencies.append(eff)
                binomial_errors.append(err)

        return np.array(bin_centers), np.array(efficiencies), np.array(binomial_errors)

    def plot_efficiency_vs_variable(
        self,
        result: RegionalQuadrantResult,
        var_name: str,
        output_dir: str | pathlib.Path,
        file_format: str = "pdf",
        bins: int = 10,
    ) -> Optional[str]:
        """Renders 2-panel scatter plot with error bars of classification efficiency vs a variable.

        Left panel: Signal Class (Photons) Efficiency vs variable.
        Right panel: Background Class (Fakes) Efficiency vs variable.

        Args:
            result: RegionalQuadrantResult containing ET, eta, predictions, and labels.
            var_name: Either 'et' or 'eta'.
            output_dir: Destination output directory.
            file_format: File format ('pdf' or 'png').
            bins: Number of equal-width bins.

        Returns:
            Saved file path or None if continuous values are unavailable.
        """
        values = result.et if var_name == "et" else result.eta
        if values is None or len(values) == 0 or not np.any(np.isfinite(values)):
            return None

        labels = result.labels
        preds1 = result.preds_strategy1
        preds2 = result.preds_strategy2

        sig_mask = labels == 1
        bg_mask = labels == 0

        s1_sig_x, s1_sig_eff, s1_sig_err = self._compute_binned_efficiency(
            values[sig_mask], preds1[sig_mask], target_class=1, bins=bins
        )
        s1_bg_x, s1_bg_eff, s1_bg_err = self._compute_binned_efficiency(
            values[bg_mask], preds1[bg_mask], target_class=0, bins=bins
        )

        s2_sig_x, s2_sig_eff, s2_sig_err = self._compute_binned_efficiency(
            values[sig_mask], preds2[sig_mask], target_class=1, bins=bins
        )
        s2_bg_x, s2_bg_eff, s2_bg_err = self._compute_binned_efficiency(
            values[bg_mask], preds2[bg_mask], target_class=0, bins=bins
        )

        if len(s1_sig_x) == 0 and len(s2_sig_x) == 0:
            return None

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        region_str = self._get_region_label(result.iet, result.ieta)
        var_label = r"Transverse Energy $E_T$ [GeV]" if var_name == "et" else r"Pseudorapidity $\eta$"

        s1_name = result.strategy1_name
        s2_name = result.strategy2_name

        # Left panel: Signal Class (Photons)
        if len(s1_sig_x) > 0:
            axes[0].errorbar(
                s1_sig_x,
                s1_sig_eff,
                yerr=s1_sig_err,
                fmt="o",
                capsize=4,
                markersize=6,
                color="#1f77b4",
                label=s1_name,
            )
        if len(s2_sig_x) > 0:
            axes[0].errorbar(
                s2_sig_x,
                s2_sig_eff,
                yerr=s2_sig_err,
                fmt="s",
                capsize=4,
                markersize=6,
                color="#ff7f0e",
                label=s2_name,
            )

        axes[0].set_xlabel(var_label, fontsize=11)
        axes[0].set_ylabel("Signal Efficiency (TPR / $P_d$)", fontsize=11)
        axes[0].set_title(f"Signal Class (Photons) Efficiency vs {var_label}", fontsize=12, weight="bold")
        axes[0].set_ylim(-0.02, 1.05)
        axes[0].grid(True, linestyle=":", alpha=0.6)
        axes[0].legend(loc="lower right", framealpha=0.9)

        # Right panel: Background Class (Fakes)
        if len(s1_bg_x) > 0:
            axes[1].errorbar(
                s1_bg_x,
                s1_bg_eff,
                yerr=s1_bg_err,
                fmt="o",
                capsize=4,
                markersize=6,
                color="#1f77b4",
                label=s1_name,
            )
        if len(s2_bg_x) > 0:
            axes[1].errorbar(
                s2_bg_x,
                s2_bg_eff,
                yerr=s2_bg_err,
                fmt="s",
                capsize=4,
                markersize=6,
                color="#ff7f0e",
                label=s2_name,
            )

        axes[1].set_xlabel(var_label, fontsize=11)
        axes[1].set_ylabel(r"Background Efficiency (TNR / $\mathrm{Eff}_{\mathrm{bg}}$)", fontsize=11)
        axes[1].set_title(f"Background Class (Fakes) Efficiency vs {var_label}", fontsize=12, weight="bold")
        axes[1].set_ylim(-0.02, 1.05)
        axes[1].grid(True, linestyle=":", alpha=0.6)
        axes[1].legend(loc="lower right", framealpha=0.9)

        fig.suptitle(
            f"Classification Efficiency vs {var_label}\n{region_str} ({s1_name} vs {s2_name})",
            fontsize=13,
            weight="bold",
        )
        plt.tight_layout()

        prefix = f"iet{result.iet}_ieta{result.ieta}" if result.iet >= 0 else "global"
        filename = f"{prefix}_efficiency_vs_{var_name}.{file_format}"
        return self.save_figure(fig, output_dir, "EfficiencyCurves", filename, file_format=file_format)

    def plot_regional_summary(
        self,
        results: List[RegionalQuadrantResult],
        output_dir: str | pathlib.Path,
        file_format: str = "pdf",
    ) -> Optional[str]:
        """Renders grouped bar plot comparing quadrant breakdown across all evaluated regions.

        Args:
            results: List of RegionalQuadrantResult objects.
            output_dir: Base directory path.
            file_format: Format ('pdf' or 'png').

        Returns:
            Saved file path or None if fewer than 2 regions.
        """
        filtered_results = [r for r in results if r.iet >= 0]
        if len(filtered_results) < 2:
            return None

        regions = [f"iet{r.iet}.ieta{r.ieta}" for r in filtered_results]
        s1_name = filtered_results[0].strategy1_name
        s2_name = filtered_results[0].strategy2_name

        both_correct = np.array([r.overall_metrics.both_correct_ratio for r in filtered_results])
        s1_only = np.array([r.overall_metrics.model1_only_correct_ratio for r in filtered_results])
        s2_only = np.array([r.overall_metrics.model2_only_correct_ratio for r in filtered_results])
        both_wrong = np.array([r.overall_metrics.both_wrong_ratio for r in filtered_results])

        fig_width = max(12, len(regions) * 1.6)
        fig, ax = plt.subplots(figsize=(fig_width, 6.5))
        x = np.arange(len(regions))
        bar_width = 0.20

        ax.bar(x - 1.5 * bar_width, both_correct, bar_width, label="Both Correct", color="#2ca02c")
        ax.bar(x - 0.5 * bar_width, s1_only, bar_width, label=f"{s1_name} Only", color="#1f77b4")
        ax.bar(x + 0.5 * bar_width, s2_only, bar_width, label=f"{s2_name} Only", color="#ff7f0e")
        ax.bar(x + 1.5 * bar_width, both_wrong, bar_width, label="Both Wrong", color="#d62728")

        ax.set_ylabel("Proportion of Holdout Events", fontsize=12)
        ax.set_title(
            f"Regional Quadrant Distribution Summary\n{s1_name} vs {s2_name}",
            fontsize=14,
            weight="bold",
        )
        ax.set_xticks(x)
        ax.set_xticklabels(regions, rotation=45, ha="right", fontsize=10)
        ax.set_ylim(0.0, 1.05)
        ax.legend(loc="upper right", framealpha=0.9)
        ax.grid(axis="y", linestyle=":", alpha=0.6)

        plt.tight_layout()
        filename = f"regional_quadrant_summary.{file_format}"
        return self.save_figure(fig, output_dir, "Summary", filename, file_format=file_format)

    def plot_regional_class_efficiencies(
        self,
        results: List[RegionalQuadrantResult],
        output_dir: str | pathlib.Path,
        file_format: str = "pdf",
    ) -> Optional[str]:
        """Renders grouped bar plot comparing classification efficiency per class across regions.

        Args:
            results: List of RegionalQuadrantResult objects.
            output_dir: Base directory path.
            file_format: Format ('pdf' or 'png').

        Returns:
            Saved file path or None if fewer than 2 regions.
        """
        filtered_results = [r for r in results if r.iet >= 0]
        if len(filtered_results) < 2:
            return None

        regions = [f"iet{r.iet}.ieta{r.ieta}" for r in filtered_results]
        s1_name = filtered_results[0].strategy1_name
        s2_name = filtered_results[0].strategy2_name

        s1_sig_eff = np.array([r.metrics_strategy1.pd for r in filtered_results])
        s1_sig_err = np.array([r.metrics_strategy1.sig_eff_uncertainty for r in filtered_results])
        s2_sig_eff = np.array([r.metrics_strategy2.pd for r in filtered_results])
        s2_sig_err = np.array([r.metrics_strategy2.sig_eff_uncertainty for r in filtered_results])

        s1_bg_eff = np.array([r.metrics_strategy1.bg_eff for r in filtered_results])
        s1_bg_err = np.array([r.metrics_strategy1.bg_eff_uncertainty for r in filtered_results])
        s2_bg_eff = np.array([r.metrics_strategy2.bg_eff for r in filtered_results])
        s2_bg_err = np.array([r.metrics_strategy2.bg_eff_uncertainty for r in filtered_results])

        fig_width = max(14, len(regions) * 1.8)
        fig, axes = plt.subplots(1, 2, figsize=(fig_width, 6.5))
        x = np.arange(len(regions))
        bar_width = 0.35

        # Signal Class (Photons)
        axes[0].bar(
            x - 0.5 * bar_width,
            s1_sig_eff,
            bar_width,
            yerr=s1_sig_err,
            capsize=4,
            label=s1_name,
            color="#1f77b4",
        )
        axes[0].bar(
            x + 0.5 * bar_width,
            s2_sig_eff,
            bar_width,
            yerr=s2_sig_err,
            capsize=4,
            label=s2_name,
            color="#ff7f0e",
        )
        axes[0].set_ylabel("Signal Efficiency (TPR / $P_d$)", fontsize=11)
        axes[0].set_title("Signal Class (Photons) Efficiency", fontsize=12, weight="bold")
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(regions, rotation=45, ha="right", fontsize=9)
        axes[0].set_ylim(0.0, 1.05)
        axes[0].grid(axis="y", linestyle=":", alpha=0.6)
        axes[0].legend(loc="lower right", framealpha=0.9)

        # Background Class (Fakes)
        axes[1].bar(
            x - 0.5 * bar_width,
            s1_bg_eff,
            bar_width,
            yerr=s1_bg_err,
            capsize=4,
            label=s1_name,
            color="#1f77b4",
        )
        axes[1].bar(
            x + 0.5 * bar_width,
            s2_bg_eff,
            bar_width,
            yerr=s2_bg_err,
            capsize=4,
            label=s2_name,
            color="#ff7f0e",
        )
        axes[1].set_ylabel(r"Background Efficiency (TNR / $\mathrm{Eff}_{\mathrm{bg}}$)", fontsize=11)
        axes[1].set_title("Background Class (Fakes) Efficiency", fontsize=12, weight="bold")
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(regions, rotation=45, ha="right", fontsize=9)
        axes[1].set_ylim(0.0, 1.05)
        axes[1].grid(axis="y", linestyle=":", alpha=0.6)
        axes[1].legend(loc="lower right", framealpha=0.9)

        fig.suptitle(
            f"Regional Classification Efficiencies by Class\n{s1_name} vs {s2_name}",
            fontsize=13,
            weight="bold",
        )
        plt.tight_layout()
        filename = f"regional_class_efficiencies.{file_format}"
        return self.save_figure(fig, output_dir, "Summary", filename, file_format=file_format)

    def plot(
        self,
        results: List[RegionalQuadrantResult],
        output_dir: str | pathlib.Path,
        file_format: str = "pdf",
        **kwargs: object,
    ) -> Dict[str, List[str]]:
        """Coordinates rendering of shower shape histograms, efficiency curves, and regional summaries.

        Args:
            results: List of RegionalQuadrantResult instances.
            output_dir: Base output directory.
            file_format: Format ('pdf' or 'png').

        Returns:
            Dictionary mapping plot categories to lists of saved file paths.
        """
        saved_paths: Dict[str, List[str]] = {
            "histograms": [],
            "efficiency_curves": [],
            "summary": [],
            "class_efficiencies": [],
        }

        if not results:
            log.warning("No quadrant results to plot. Skipping plot generation.")
            return saved_paths

        hist_base_dir = pathlib.Path(output_dir) / "histograms"
        eff_base_dir = pathlib.Path(output_dir) / "efficiency_curves"
        summary_dir = pathlib.Path(output_dir) / "summary"

        log.info(f"Rendering quadrant figures for {len(results)} evaluated result sets into: {output_dir}")

        for res in results:
            if res.iet < 0:
                continue
            region_tag = f"iet{res.iet}.ieta{res.ieta}"
            region_hist_dir = hist_base_dir / f"iet{res.iet}_ieta{res.ieta}"
            region_eff_dir = eff_base_dir / f"iet{res.iet}_ieta{res.ieta}"

            for var_name in res.showershapes.keys():
                path = self.plot_showershape_histogram(
                    result=res,
                    var_name=var_name,
                    output_dir=region_hist_dir,
                    file_format=file_format,
                )
                if path:
                    saved_paths["histograms"].append(path)

            et_eff_path = self.plot_efficiency_vs_variable(
                result=res,
                var_name="et",
                output_dir=region_eff_dir,
                file_format=file_format,
            )
            if et_eff_path:
                saved_paths["efficiency_curves"].append(et_eff_path)

            eta_eff_path = self.plot_efficiency_vs_variable(
                result=res,
                var_name="eta",
                output_dir=region_eff_dir,
                file_format=file_format,
            )
            if eta_eff_path:
                saved_paths["efficiency_curves"].append(eta_eff_path)

            log.info(f"[{region_tag}] Generated figures in {region_hist_dir} and {region_eff_dir}")

        global_res = next((r for r in results if r.iet < 0), None)
        if global_res is not None:
            global_eff_dir = eff_base_dir / "global"
            global_et_path = self.plot_efficiency_vs_variable(
                result=global_res,
                var_name="et",
                output_dir=global_eff_dir,
                file_format=file_format,
                bins=20,
            )
            if global_et_path:
                saved_paths["efficiency_curves"].append(global_et_path)
                log.info(f"[Global] Generated full efficiency vs ET: {global_et_path}")

            global_eta_path = self.plot_efficiency_vs_variable(
                result=global_res,
                var_name="eta",
                output_dir=global_eff_dir,
                file_format=file_format,
                bins=20,
            )
            if global_eta_path:
                saved_paths["efficiency_curves"].append(global_eta_path)
                log.info(f"[Global] Generated full efficiency vs eta: {global_eta_path}")

        # Summary plots
        summary_path = self.plot_regional_summary(results, summary_dir, file_format=file_format)
        if summary_path:
            saved_paths["summary"].append(summary_path)
            log.info(f"[Summary] Generated regional grouped summary: {summary_path}")

        eff_path = self.plot_regional_class_efficiencies(results, summary_dir, file_format=file_format)
        if eff_path:
            saved_paths["class_efficiencies"].append(eff_path)
            log.info(f"[Summary] Generated regional class efficiencies: {eff_path}")

        return saved_paths
