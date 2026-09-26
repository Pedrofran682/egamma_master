import pathlib
from typing import List
import matplotlib.pyplot as plt
import numpy as np

from src.core.Validation.HoldoutEvaluator import HoldoutEvaluationResult


class EfficiencyPlotter:
    """Renders binned signal detection efficiency curves across kinematic variables.

    Attributes:
        et_index: Feature column index for transverse energy.
        eta_index: Feature column index for pseudorapidity.
        et_cutoff: Kinematic cutoff in MeV (e.g. 150000 MeV for 150 GeV).
    """

    def __init__(
        self,
        et_index: int = 1,
        eta_index: int = 2,
        et_cutoff: float = 150000.0,
    ) -> None:
        """Initializes EfficiencyPlotter with column indices and energy cutoff.

        Args:
            et_index: Column index for transverse energy.
            eta_index: Column index for pseudorapidity.
            et_cutoff: Upper cutoff for transverse energy in MeV.
        """
        self.et_index: int = et_index
        self.eta_index: int = eta_index
        self.et_cutoff: float = et_cutoff

    def plot_efficiency_vs_feature(
        self,
        feature_values: np.ndarray,
        y_holdout: np.ndarray,
        preds: np.ndarray,
        file_path: pathlib.Path | str,
        feature_name: str = "Feature",
        bins: int = 20,
        label: str = "Efficiency (Signal)",
    ) -> None:
        """Renders binned efficiency with binomial uncertainty error bars.

        Args:
            feature_values: Array of continuous feature values (e.g., ET or eta).
            y_holdout: True binary labels array.
            preds: Predicted binary classification labels.
            file_path: Output file destination path.
            feature_name: Label string for the horizontal axis.
            bins: Number of equal-width bins.
            label: Legend label text.
        """
        signal_mask = y_holdout == 1
        signal_feature = feature_values[signal_mask]
        signal_preds = preds[signal_mask]

        bin_edges = np.histogram_bin_edges(signal_feature, bins=bins)
        bin_centers, efficiencies, errors = [], [], []

        for i in range(len(bin_edges) - 1):
            bin_min = bin_edges[i]
            bin_max = bin_edges[i + 1]

            if i == len(bin_edges) - 2:
                in_bin = (signal_feature >= bin_min) & (signal_feature <= bin_max)
            else:
                in_bin = (signal_feature >= bin_min) & (signal_feature < bin_max)

            total_in_bin = np.sum(in_bin)
            if total_in_bin > 0:
                correct_preds = np.sum(signal_preds[in_bin] == 1)
                eff = correct_preds / total_in_bin
                error = np.sqrt((eff * (1 - eff)) / total_in_bin)

                bin_centers.append((bin_min + bin_max) / 2)
                efficiencies.append(eff)
                errors.append(error)

        fig = plt.figure(figsize=(8, 5))
        plt.errorbar(
            bin_centers,
            efficiencies,
            yerr=errors,
            fmt="o-",
            capsize=4,
            color="#1f77b4",
            linewidth=2,
            label=label,
        )
        plt.xlabel(f"{feature_name}", fontsize=12)
        plt.ylabel("Detection Efficiency (TPR)", fontsize=12)
        plt.title(f"Detection Efficiency vs {feature_name}", fontsize=14)
        plt.ylim(-0.05, 1.05)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend(fontsize=11)

        plt.savefig(file_path, format="png", transparent=True, bbox_inches="tight")
        plt.close(fig)

    def generate_global_plots(
        self,
        results: List[HoldoutEvaluationResult],
        plot_dir: pathlib.Path,
        config_name: str,
    ) -> None:
        """Concatenates holdout data across all regions and plots global ET and eta efficiencies.

        Args:
            results: List of HoldoutEvaluationResult objects collected across regions.
            plot_dir: Target output directory.
            config_name: Model configuration name used in filenames.
        """
        if not results:
            return

        x_h_global = np.concatenate([r.x_holdout_full for r in results], axis=0)
        y_h_global = np.concatenate([r.y_holdout for r in results], axis=0)
        preds_05_global = np.concatenate([r.preds_05 for r in results], axis=0)
        preds_cut_global = np.concatenate([r.preds_cut for r in results], axis=0)

        et_values_global = x_h_global[:, self.et_index]
        et_cut_mask = et_values_global < self.et_cutoff

        x_h_cut = x_h_global[et_cut_mask]
        y_h_cut = y_h_global[et_cut_mask]
        preds_05_cut = preds_05_global[et_cut_mask]
        preds_cut_cut = preds_cut_global[et_cut_mask]

        et_cut_values = x_h_cut[:, self.et_index] / 1000.0
        eta_cut_values = x_h_cut[:, self.eta_index]
        et_cutoff_gev = int(self.et_cutoff / 1000)

        path_et_05 = plot_dir / f"{config_name}_global_model_eff_05_et.png"
        self.plot_efficiency_vs_feature(
            et_cut_values,
            y_h_cut,
            preds_05_cut,
            path_et_05,
            feature_name=f"$E_T$ [GeV] (< {et_cutoff_gev} GeV)",
            bins=10,
            label="Efficiency (Threshold 0.5)",
        )

        path_et_cut = plot_dir / f"{config_name}_global_model_eff_cut_et.png"
        self.plot_efficiency_vs_feature(
            et_cut_values,
            y_h_cut,
            preds_cut_cut,
            path_et_cut,
            feature_name=f"$E_T$ [GeV] (< {et_cutoff_gev} GeV) (Proposed Cut)",
            bins=10,
            label="Efficiency (Proposed Cut)",
        )

        path_eta_05 = plot_dir / f"{config_name}_global_model_eff_05_eta.png"
        self.plot_efficiency_vs_feature(
            eta_cut_values,
            y_h_cut,
            preds_05_cut,
            path_eta_05,
            feature_name=f"$\\eta$ (for $E_T$ < {et_cutoff_gev} GeV)",
            bins=20,
            label="Efficiency (Threshold 0.5)",
        )

        path_eta_cut = plot_dir / f"{config_name}_global_model_eff_cut_eta.png"
        self.plot_efficiency_vs_feature(
            eta_cut_values,
            y_h_cut,
            preds_cut_cut,
            path_eta_cut,
            feature_name=f"$\\eta$ (for $E_T$ < {et_cutoff_gev} GeV) (Proposed Cut)",
            bins=20,
            label="Efficiency (Proposed Cut)",
        )
