import logging
import os
import pathlib
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from src.core.Interfaces.BasePlotter import BasePlotter
from src.core.Plotting.RegionDistributionPlotter import DEFAULT_ET_INTERVALS, DEFAULT_ETA_INTERVALS
from src.core.Validation.QuadrantAnalyzer import QuadrantMetrics, RegionalQuadrantResult
from src.utils import create_folder

log = logging.getLogger(__name__)


class QuadrantPlotter(BasePlotter):
    """Generates score scatter plots and regional grouped summary bar plots for quadrant analysis."""

    def __init__(
        self,
        et_intervals: Optional[Dict[int, str]] = None,
        eta_intervals: Optional[Dict[int, str]] = None,
    ) -> None:
        """Initializes the QuadrantPlotter with optional kinematic label intervals.

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

    def plot_score_scatter(
        self,
        result: RegionalQuadrantResult,
        output_dir: str | pathlib.Path,
        file_format: str = "pdf",
        max_points: int = 5000,
    ) -> str:
        """Renders score scatter comparison in model probability space partitioned by quadrant agreement.

        Args:
            result: RegionalQuadrantResult containing scores, true labels, and thresholds.
            output_dir: Target destination directory.
            file_format: Format ('pdf' or 'png').
            max_points: Maximum subsampled points to avoid graphic rendering slowdown.

        Returns:
            Saved file path.
        """
        n_points = len(result.labels)
        if n_points > max_points:
            indices = np.random.choice(n_points, size=max_points, replace=False)
            p1 = result.probs_model1[indices]
            p2 = result.probs_model2[indices]
            y = result.labels[indices]
        else:
            p1 = result.probs_model1
            p2 = result.probs_model2
            y = result.labels

        th1 = result.threshold_model1
        th2 = result.threshold_model2

        pred1 = (p1 > th1).astype(int)
        pred2 = (p2 > th2).astype(int)
        c1 = pred1 == y
        c2 = pred2 == y

        categories = np.empty(len(y), dtype=object)
        categories[c1 & c2] = "Both Correct"
        categories[c1 & ~c2] = f"{result.model1_name} Only"
        categories[~c1 & c2] = f"{result.model2_name} Only"
        categories[~c1 & ~c2] = "Both Wrong"

        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        region_str = self._get_region_label(result.iet, result.ieta)
        fig.suptitle(
            f"Model Score Distribution & Quadrant Agreement — {region_str}\n"
            f"{result.model1_name} vs {result.model2_name}",
            fontsize=14,
            weight="bold",
        )

        signal_mask = y == 1
        axes[0].scatter(
            p1[~signal_mask],
            p2[~signal_mask],
            alpha=0.35,
            s=12,
            color="#d62728",
            label=f"Background (0) ({np.sum(~signal_mask):,})",
            rasterized=True,
        )
        axes[0].scatter(
            p1[signal_mask],
            p2[signal_mask],
            alpha=0.35,
            s=12,
            color="#2ca02c",
            label=f"Signal (1) ({np.sum(signal_mask):,})",
            rasterized=True,
        )
        axes[0].set_xlabel(f"{result.model1_name} Score", fontsize=11)
        axes[0].set_ylabel(f"{result.model2_name} Score", fontsize=11)
        axes[0].set_title("Events by Ground Truth Class", fontsize=12)
        axes[0].grid(True, linestyle=":", alpha=0.5)

        palette = {
            "Both Correct": "#2ca02c",
            f"{result.model1_name} Only": "#1f77b4",
            f"{result.model2_name} Only": "#ff7f0e",
            "Both Wrong": "#d62728",
        }
        for cat, color in palette.items():
            mask = categories == cat
            if np.any(mask):
                axes[1].scatter(
                    p1[mask],
                    p2[mask],
                    alpha=0.45,
                    s=14,
                    color=color,
                    label=f"{cat} ({np.sum(mask):,})",
                    rasterized=True,
                )

        axes[1].set_xlabel(f"{result.model1_name} Score", fontsize=11)
        axes[1].set_ylabel(f"{result.model2_name} Score", fontsize=11)
        axes[1].set_title("Events by Classification Agreement", fontsize=12)
        axes[1].grid(True, linestyle=":", alpha=0.5)

        for ax in axes:
            ax.axvline(
                x=th1,
                color="black",
                linestyle="--",
                linewidth=1.2,
                alpha=0.8,
                label=f"Threshold 1 ({th1:.3f})",
            )
            ax.axhline(
                y=th2,
                color="black",
                linestyle="--",
                linewidth=1.2,
                alpha=0.8,
                label=f"Threshold 2 ({th2:.3f})",
            )
            ax.set_xlim(-0.02, 1.02)
            ax.set_ylim(-0.02, 1.02)
            ax.legend(loc="upper left", framealpha=0.9, fontsize=9)

        plt.tight_layout()
        prefix = f"iet{result.iet}_ieta{result.ieta}" if result.iet >= 0 else "global"
        filename = f"{prefix}_quadrant_score_scatter.{file_format}"
        return self.save_figure(fig, output_dir, "ScoreScatter", filename, file_format=file_format)

    def plot_kinematic_scatter(
        self,
        result: RegionalQuadrantResult,
        output_dir: str | pathlib.Path,
        file_format: str = "pdf",
        max_points: int = 5000,
    ) -> str:
        """Renders scatter comparison in (eta, ET) kinematic space partitioned by quadrant agreement.

        Args:
            result: RegionalQuadrantResult containing scores, true labels, ET, eta, and thresholds.
            output_dir: Target destination directory.
            file_format: Format ('pdf' or 'png').
            max_points: Max subsampled points to avoid graphic rendering slowdown.

        Returns:
            Saved file path.
        """
        n_points = len(result.labels)
        eta = result.eta if result.eta is not None and len(result.eta) == n_points else np.zeros(n_points)
        et = result.et if result.et is not None and len(result.et) == n_points else np.zeros(n_points)

        if n_points > max_points:
            indices = np.random.choice(n_points, size=max_points, replace=False)
            p1 = result.probs_model1[indices]
            p2 = result.probs_model2[indices]
            y = result.labels[indices]
            eta = eta[indices]
            et = et[indices]
        else:
            p1 = result.probs_model1
            p2 = result.probs_model2
            y = result.labels

        th1 = result.threshold_model1
        th2 = result.threshold_model2

        pred1 = (p1 > th1).astype(int)
        pred2 = (p2 > th2).astype(int)
        c1 = pred1 == y
        c2 = pred2 == y

        categories = np.empty(len(y), dtype=object)
        categories[c1 & c2] = "Both Correct"
        categories[c1 & ~c2] = f"{result.model1_name} Only"
        categories[~c1 & c2] = f"{result.model2_name} Only"
        categories[~c1 & ~c2] = "Both Wrong"

        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        region_str = self._get_region_label(result.iet, result.ieta)
        fig.suptitle(
            f"Kinematic Distribution & Quadrant Agreement — {region_str}\n"
            f"{result.model1_name} vs {result.model2_name}",
            fontsize=14,
            weight="bold",
        )

        signal_mask = y == 1
        axes[0].scatter(
            eta[~signal_mask],
            et[~signal_mask],
            alpha=0.35,
            s=12,
            color="#d62728",
            label=f"Background (0) ({np.sum(~signal_mask):,})",
            rasterized=True,
        )
        axes[0].scatter(
            eta[signal_mask],
            et[signal_mask],
            alpha=0.35,
            s=12,
            color="#2ca02c",
            label=f"Signal (1) ({np.sum(signal_mask):,})",
            rasterized=True,
        )
        axes[0].set_xlabel(r"Pseudorapidity $\eta$", fontsize=11)
        axes[0].set_ylabel(r"Transverse Energy $E_T$ [GeV]", fontsize=11)
        axes[0].set_title("Events by Ground Truth Class", fontsize=12)
        axes[0].legend(loc="upper right", framealpha=0.9)
        axes[0].grid(True, linestyle=":", alpha=0.5)

        palette = {
            "Both Correct": "#2ca02c",
            f"{result.model1_name} Only": "#1f77b4",
            f"{result.model2_name} Only": "#ff7f0e",
            "Both Wrong": "#d62728",
        }
        for cat, color in palette.items():
            mask = categories == cat
            if np.any(mask):
                axes[1].scatter(
                    eta[mask],
                    et[mask],
                    alpha=0.45,
                    s=14,
                    color=color,
                    label=f"{cat} ({np.sum(mask):,})",
                    rasterized=True,
                )

        axes[1].set_xlabel(r"Pseudorapidity $\eta$", fontsize=11)
        axes[1].set_ylabel(r"Transverse Energy $E_T$ [GeV]", fontsize=11)
        axes[1].set_title("Events by Classification Agreement", fontsize=12)
        axes[1].legend(loc="upper right", framealpha=0.9)
        axes[1].grid(True, linestyle=":", alpha=0.5)

        if len(eta) > 0 and (np.max(eta) > np.min(eta) or np.max(et) > np.min(et)):
            eta_min, eta_max = float(np.min(eta)), float(np.max(eta))
            et_min, et_max = float(np.min(et)), float(np.max(et))
            eta_pad = max((eta_max - eta_min) * 0.05, 0.02)
            et_pad = max((et_max - et_min) * 0.05, 0.5)
            for ax in axes:
                ax.set_xlim(eta_min - eta_pad, eta_max + eta_pad)
                ax.set_ylim(max(0.0, et_min - et_pad), et_max + et_pad)

        plt.tight_layout()
        prefix = f"iet{result.iet}_ieta{result.ieta}" if result.iet >= 0 else "global"
        filename = f"{prefix}_quadrant_region_scatter.{file_format}"
        return self.save_figure(fig, output_dir, "RegionScatter", filename, file_format=file_format)

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
        m1_name = filtered_results[0].model1_name
        m2_name = filtered_results[0].model2_name

        both_correct = np.array([r.overall_metrics.both_correct_ratio for r in filtered_results])
        m1_only = np.array([r.overall_metrics.model1_only_correct_ratio for r in filtered_results])
        m2_only = np.array([r.overall_metrics.model2_only_correct_ratio for r in filtered_results])
        both_wrong = np.array([r.overall_metrics.both_wrong_ratio for r in filtered_results])

        fig_width = max(12, len(regions) * 1.6)
        fig, ax = plt.subplots(figsize=(fig_width, 6.5))
        x = np.arange(len(regions))
        bar_width = 0.20

        ax.bar(x - 1.5 * bar_width, both_correct, bar_width, label="Both Correct", color="#2ca02c")
        ax.bar(x - 0.5 * bar_width, m1_only, bar_width, label=f"{m1_name} Only", color="#1f77b4")
        ax.bar(x + 0.5 * bar_width, m2_only, bar_width, label=f"{m2_name} Only", color="#ff7f0e")
        ax.bar(x + 1.5 * bar_width, both_wrong, bar_width, label="Both Wrong", color="#d62728")

        ax.set_ylabel("Proportion of Holdout Events", fontsize=12)
        ax.set_title(
            f"Regional Quadrant Distribution Summary\n{m1_name} vs {m2_name}",
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

    def plot(
        self,
        results: List[RegionalQuadrantResult],
        output_dir: str | pathlib.Path,
        file_format: str = "pdf",
        **kwargs: object,
    ) -> Dict[str, List[str]]:
        """Coordinates rendering of all quadrant analysis figures into scores and regions folders.

        Args:
            results: List of RegionalQuadrantResult instances.
            output_dir: Base output directory.
            file_format: Format ('pdf' or 'png').

        Returns:
            Dictionary mapping plot categories to lists of saved file paths.
        """
        saved_paths: Dict[str, List[str]] = {
            "score_scatters": [],
            "region_scatters": [],
            "summary": [],
        }

        if not results:
            log.warning("No quadrant results to plot. Skipping plot generation.")
            return saved_paths

        scores_dir = pathlib.Path(output_dir) / "scores"
        regions_dir = pathlib.Path(output_dir) / "regions"

        log.info(f"Rendering quadrant figures for {len(results)} evaluated result sets into: {output_dir}")

        for res in results:
            if res.iet < 0:
                continue
            region_tag = f"iet{res.iet}.ieta{res.ieta}"
            score_path = self.plot_score_scatter(res, scores_dir, file_format=file_format)
            saved_paths["score_scatters"].append(score_path)
            log.info(f"[{region_tag}] Generated score scatter plot: {score_path}")

            region_path = self.plot_kinematic_scatter(res, regions_dir, file_format=file_format)
            saved_paths["region_scatters"].append(region_path)
            log.info(f"[{region_tag}] Generated region scatter plot: {region_path}")

        summary_path = self.plot_regional_summary(results, regions_dir, file_format=file_format)
        if summary_path:
            saved_paths["summary"].append(summary_path)
            log.info(f"[Summary] Generated regional grouped summary: {summary_path}")

        return saved_paths
