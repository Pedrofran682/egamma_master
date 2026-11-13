from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
from scipy import stats


class PileupAnalyzer:
    """Statistical engine for pileup sensitivity, efficiency drift, and vertex correlation."""

    @staticmethod
    def extract_feature(
        features: np.ndarray,
        feature_names: np.ndarray,
        target_name: str,
    ) -> np.ndarray:
        """Extracts column values or computes derived physics variables.

        Args:
            features: 2D array of event features (N, D).
            feature_names: Array of feature name strings.
            target_name: Target variable name.

        Returns:
            1D numpy array of values.
        """
        name_to_idx = {name.strip(): i for i, name in enumerate(feature_names)}

        if target_name.lower() in ["eratio", "eration", "l2_eratio"]:
            if "trig_L2_calo_emaxs1" in name_to_idx and "trig_L2_calo_e2tsts1" in name_to_idx:
                emaxs1 = features[:, name_to_idx["trig_L2_calo_emaxs1"]]
                e2tsts1 = features[:, name_to_idx["trig_L2_calo_e2tsts1"]]
                denom = emaxs1 + e2tsts1
                safe_denom = np.where(denom != 0.0, denom, 1.0)
                return np.where(denom != 0.0, (emaxs1 - e2tsts1) / safe_denom, 0.0)
            raise KeyError("Cannot compute Eratio: required sub-features missing.")

        if target_name in name_to_idx:
            return features[:, name_to_idx[target_name]]

        for name, idx in name_to_idx.items():
            if target_name.lower() == name.lower():
                return features[:, idx]

        raise KeyError(f"Feature '{target_name}' not found in feature_names.")

    @classmethod
    def compute_efficiency_vs_mu(
        cls,
        mu_values: np.ndarray,
        y_true: np.ndarray,
        predictions: np.ndarray,
        bins: Union[int, np.ndarray] = 10,
        bin_range: Optional[Tuple[float, float]] = None,
        is_signal: bool = True,
    ) -> Dict[str, np.ndarray]:
        """Calculates binned detection efficiency or fake rate and binomial errors vs mu.

        Args:
            mu_values: 1D array of avgmu values.
            y_true: Ground truth binary labels (0 or 1).
            predictions: Predicted binary classifications (0 or 1).
            bins: Number of bins or explicit bin edges.
            bin_range: Optional tuple of (low, high) range for binning.
            is_signal: True for signal efficiency (PD), False for background fake rate (PF).

        Returns:
            Dictionary with bin edges, centers, counts, efficiencies, and uncertainties.
        """
        target_label = 1 if is_signal else 0
        mask = (y_true == target_label) & np.isfinite(mu_values)

        mu_sel = mu_values[mask]
        preds_sel = predictions[mask]

        if len(mu_sel) == 0:
            raise ValueError("No valid events found for requested class selection.")

        if isinstance(bins, int):
            if bin_range is not None:
                low, high = float(bin_range[0]), float(bin_range[1])
            else:
                low = float(np.min(mu_sel))
                high = float(np.max(mu_sel))
            if low == high:
                low -= 1.0
                high += 1.0
            bin_edges = np.linspace(low, high, bins + 1)
        else:
            bin_edges = np.asarray(bins, dtype=float)

        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        n_bins = len(bin_centers)

        total_counts = np.zeros(n_bins, dtype=int)
        pass_counts = np.zeros(n_bins, dtype=int)
        efficiencies = np.full(n_bins, np.nan, dtype=float)
        uncertainties = np.zeros(n_bins, dtype=float)

        for i in range(n_bins):
            in_bin = (mu_sel >= bin_edges[i]) & (
                (mu_sel < bin_edges[i + 1]) if i < n_bins - 1 else (mu_sel <= bin_edges[i + 1])
            )
            n_tot = int(np.sum(in_bin))
            total_counts[i] = n_tot

            if n_tot > 0:
                n_pass = int(np.sum(preds_sel[in_bin] == 1))
                pass_counts[i] = n_pass
                eff = n_pass / n_tot
                efficiencies[i] = eff
                uncertainties[i] = np.sqrt(eff * (1.0 - eff) / n_tot)

        return {
            "bin_edges": bin_edges,
            "bin_centers": bin_centers,
            "total_counts": total_counts,
            "pass_counts": pass_counts,
            "efficiencies": efficiencies,
            "uncertainties": uncertainties,
        }

    @classmethod
    def compute_profile_vs_mu(
        cls,
        mu_values: np.ndarray,
        observable_values: np.ndarray,
        bins: Union[int, np.ndarray] = 10,
        bin_range: Optional[Tuple[float, float]] = None,
    ) -> Dict[str, np.ndarray]:
        """Computes mean profile and standard error of the mean for an observable vs mu.

        Args:
            mu_values: 1D array of avgmu values.
            observable_values: 1D array of observable values.
            bins: Number of bins or explicit bin edges.
            bin_range: Optional tuple of (low, high) range for binning.

        Returns:
            Dictionary with bin edges, centers, counts, means, stds, and sems.
        """
        valid_mask = np.isfinite(mu_values) & np.isfinite(observable_values)
        mu_clean = mu_values[valid_mask]
        obs_clean = observable_values[valid_mask]

        if len(mu_clean) == 0:
            raise ValueError("No valid events found for observable profile calculation.")

        if isinstance(bins, int):
            if bin_range is not None:
                low, high = float(bin_range[0]), float(bin_range[1])
            else:
                low = float(np.min(mu_clean))
                high = float(np.max(mu_clean))
            if low == high:
                low -= 1.0
                high += 1.0
            bin_edges = np.linspace(low, high, bins + 1)
        else:
            bin_edges = np.asarray(bins, dtype=float)

        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        n_bins = len(bin_centers)

        counts = np.zeros(n_bins, dtype=int)
        means = np.full(n_bins, np.nan, dtype=float)
        stds = np.zeros(n_bins, dtype=float)
        sems = np.zeros(n_bins, dtype=float)

        for i in range(n_bins):
            in_bin = (mu_clean >= bin_edges[i]) & (
                (mu_clean < bin_edges[i + 1]) if i < n_bins - 1 else (mu_clean <= bin_edges[i + 1])
            )
            bin_obs = obs_clean[in_bin]
            n = len(bin_obs)
            counts[i] = n

            if n > 0:
                mean_val = float(np.mean(bin_obs))
                means[i] = mean_val
                std_val = float(np.std(bin_obs, ddof=1)) if n > 1 else 0.0
                stds[i] = std_val
                sems[i] = std_val / np.sqrt(n)

        return {
            "bin_edges": bin_edges,
            "bin_centers": bin_centers,
            "counts": counts,
            "means": means,
            "stds": stds,
            "sems": sems,
        }

    @classmethod
    def compute_vertex_correlation(
        cls,
        mu_values: np.ndarray,
        npv_values: np.ndarray,
        bins: Union[int, Tuple[int, int]] = 30,
    ) -> Dict[str, Any]:
        """Computes 2D density grid, mean profile, and linear regression between NPV and mu.

        Args:
            mu_values: 1D array of avgmu values.
            npv_values: 1D array of primary vertex counts.
            bins: Bin count for 2D histogram.

        Returns:
            Dictionary containing regression parameters, 2D histogram data, and 1D profile.
        """
        valid_mask = np.isfinite(mu_values) & np.isfinite(npv_values)
        mu_clean = mu_values[valid_mask]
        npv_clean = npv_values[valid_mask]

        if len(mu_clean) < 2:
            raise ValueError("Insufficient valid events for vertex correlation analysis.")

        reg = stats.linregress(mu_clean, npv_clean)

        if isinstance(bins, int):
            bins_2d = (bins, bins)
        else:
            bins_2d = bins

        mu_low, mu_high = np.percentile(mu_clean, [0.5, 99.5])
        npv_low, npv_high = np.percentile(npv_clean, [0.5, 99.5])
        range_2d = [[mu_low, mu_high], [max(0.0, npv_low), npv_high]]

        hist_counts, xedges, yedges = np.histogram2d(
            mu_clean, npv_clean, bins=bins_2d, range=range_2d
        )

        profile = cls.compute_profile_vs_mu(mu_clean, npv_clean, bins=15)

        return {
            "slope": float(reg.slope),
            "intercept": float(reg.intercept),
            "r_value": float(reg.rvalue),
            "r_squared": float(reg.rvalue**2),
            "p_value": float(reg.pvalue),
            "std_err": float(reg.stderr),
            "hist2d_counts": hist_counts,
            "hist2d_xedges": xedges,
            "hist2d_yedges": yedges,
            "profile": profile,
        }
