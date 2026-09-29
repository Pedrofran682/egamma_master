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
        signal_feature_values = feature_values[signal_mask]
        signal_predictions = preds[signal_mask]

        bin_edges = np.histogram_bin_edges(signal_feature_values, bins=bins)
        bin_centers: List[float] = []
        efficiencies: List[float] = []
        binomial_uncertainties: List[float] = []

        for bin_index in range(len(bin_edges) - 1):
            bin_lower = bin_edges[bin_index]
            bin_upper = bin_edges[bin_index + 1]

            if bin_index == len(bin_edges) - 2:
                in_bin_mask = (signal_feature_values >= bin_lower) & (signal_feature_values <= bin_upper)
            else:
                in_bin_mask = (signal_feature_values >= bin_lower) & (signal_feature_values < bin_upper)

            total_events_in_bin = int(np.sum(in_bin_mask))
            if total_events_in_bin > 0:
                true_positive_count = int(np.sum(signal_predictions[in_bin_mask] == 1))
                efficiency = true_positive_count / total_events_in_bin
                binomial_uncertainty = float(np.sqrt((efficiency * (1 - efficiency)) / total_events_in_bin))

                bin_centers.append((bin_lower + bin_upper) / 2.0)
                efficiencies.append(efficiency)
                binomial_uncertainties.append(binomial_uncertainty)

        figure = plt.figure(figsize=(8, 5))
        plt.errorbar(
            bin_centers,
            efficiencies,
            yerr=binomial_uncertainties,
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
        plt.close(figure)

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

        global_features = np.concatenate([evaluation_result.holdout_features_full for evaluation_result in results], axis=0)
        global_targets = np.concatenate([evaluation_result.holdout_labels for evaluation_result in results], axis=0)
        global_predictions_default = np.concatenate([evaluation_result.predictions_default_threshold for evaluation_result in results], axis=0)
        global_predictions_tuned = np.concatenate([evaluation_result.predictions_calibrated_cut for evaluation_result in results], axis=0)

        global_transverse_energies = global_features[:, self.et_index]
        energy_filter_mask = global_transverse_energies < self.et_cutoff

        filtered_features = global_features[energy_filter_mask]
        filtered_targets = global_targets[energy_filter_mask]
        filtered_predictions_default = global_predictions_default[energy_filter_mask]
        filtered_predictions_tuned_cut = global_predictions_tuned[energy_filter_mask]

        filtered_transverse_energy_gev = filtered_features[:, self.et_index] / 1000.0
        filtered_pseudorapidity_values = filtered_features[:, self.eta_index]
        et_cutoff_gev = int(self.et_cutoff / 1000)

        et_standard_cut_plot_path = plot_dir / f"{config_name}_signal_efficiency_vs_et_standard_cut.png"
        self.plot_efficiency_vs_feature(
            filtered_transverse_energy_gev,
            filtered_targets,
            filtered_predictions_default,
            et_standard_cut_plot_path,
            feature_name=f"$E_T$ [GeV] (< {et_cutoff_gev} GeV)",
            bins=10,
            label="Efficiency (Threshold 0.5)",
        )

        et_target_pd_cut_plot_path = plot_dir / f"{config_name}_signal_efficiency_vs_et_target_pd_cut.png"
        self.plot_efficiency_vs_feature(
            filtered_transverse_energy_gev,
            filtered_targets,
            filtered_predictions_tuned_cut,
            et_target_pd_cut_plot_path,
            feature_name=f"$E_T$ [GeV] (< {et_cutoff_gev} GeV) (Proposed Cut)",
            bins=10,
            label="Efficiency (Proposed Cut)",
        )

        eta_standard_cut_plot_path = plot_dir / f"{config_name}_signal_efficiency_vs_eta_standard_cut.png"
        self.plot_efficiency_vs_feature(
            filtered_pseudorapidity_values,
            filtered_targets,
            filtered_predictions_default,
            eta_standard_cut_plot_path,
            feature_name=f"$\\eta$ (for $E_T$ < {et_cutoff_gev} GeV)",
            bins=20,
            label="Efficiency (Threshold 0.5)",
        )

        eta_target_pd_cut_plot_path = plot_dir / f"{config_name}_signal_efficiency_vs_eta_target_pd_cut.png"
        self.plot_efficiency_vs_feature(
            filtered_pseudorapidity_values,
            filtered_targets,
            filtered_predictions_tuned_cut,
            eta_target_pd_cut_plot_path,
            feature_name=f"$\\eta$ (for $E_T$ < {et_cutoff_gev} GeV) (Proposed Cut)",
            bins=20,
            label="Efficiency (Proposed Cut)",
        )
