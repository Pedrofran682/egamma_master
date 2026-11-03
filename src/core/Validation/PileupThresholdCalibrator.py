from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from scipy import stats


@dataclass
class OperatingPointCalibration:
    """Encapsulates calibration parameters and fit metrics for a single working point."""

    working_point_name: str
    target_pd: float
    mu_bin_centers: np.ndarray
    slice_thresholds: np.ndarray
    slice_threshold_errors: np.ndarray
    slope: float
    intercept: float
    r_squared: float
    p_value: float
    std_err: float


@dataclass
class Pileup2DDistribution:
    """Encapsulates the 2D joint distribution data of discriminant vs avgmu."""

    hist2d_counts: np.ndarray
    hist2d_xedges: np.ndarray
    hist2d_yedges: np.ndarray
    mu_values: np.ndarray
    discriminant_values: np.ndarray


class PileupThresholdCalibrator:
    """Calculates pileup-dependent decision boundaries T(mu) = a*mu + b across operating points."""

    def __init__(
        self,
        bins_mu: Union[int, np.ndarray] = 10,
        mu_range: Optional[Tuple[float, float]] = None,
        n_bootstraps: int = 100,
        random_seed: int = 42,
    ) -> None:
        """Initializes PileupThresholdCalibrator with binning and bootstrap configurations.

        Args:
            bins_mu: Number of mu bins or explicit bin edges array.
            mu_range: Optional min and max range for mu binning.
            n_bootstraps: Number of bootstrap iterations for threshold uncertainty.
            random_seed: Seed for reproducible bootstrap sampling.
        """
        self.bins_mu: Union[int, np.ndarray] = bins_mu
        self.mu_range: Optional[Tuple[float, float]] = mu_range
        self.n_bootstraps: int = n_bootstraps
        self.random_seed: int = random_seed

    def compute_2d_distribution(
        self,
        mu_values: np.ndarray,
        discriminant_values: np.ndarray,
        bins: Union[int, Tuple[int, int]] = 35,
    ) -> Pileup2DDistribution:
        """Generates 2D histogram grid of discriminant values versus avgmu.

        Args:
            mu_values: 1D array of avgmu values for signal events.
            discriminant_values: 1D array of raw neural network outputs for signal events.
            bins: Bin count for 2D histogram or tuple of (mu_bins, disc_bins).

        Returns:
            Pileup2DDistribution containing histogram counts, edges, and clean arrays.
        """
        valid_mask = np.isfinite(mu_values) & np.isfinite(discriminant_values)
        mu_clean = mu_values[valid_mask]
        disc_clean = discriminant_values[valid_mask]

        if len(mu_clean) == 0:
            raise ValueError("No valid finite events found for 2D distribution computation.")

        if isinstance(bins, int):
            bins_2d = (bins, bins)
        else:
            bins_2d = bins

        mu_low, mu_high = np.percentile(mu_clean, [0.5, 99.5])
        disc_low, disc_high = np.percentile(disc_clean, [0.5, 99.5])
        range_2d = [[mu_low, mu_high], [disc_low, disc_high]]

        hist_counts, xedges, yedges = np.histogram2d(
            mu_clean, disc_clean, bins=bins_2d, range=range_2d
        )

        return Pileup2DDistribution(
            hist2d_counts=hist_counts,
            hist2d_xedges=xedges,
            hist2d_yedges=yedges,
            mu_values=mu_clean,
            discriminant_values=disc_clean,
        )

    def calibrate_working_point(
        self,
        mu_values: np.ndarray,
        discriminant_values: np.ndarray,
        target_pd: float,
        working_point_name: str = "",
    ) -> OperatingPointCalibration:
        """Extracts slice thresholds and performs linear fit for a target detection probability.

        Args:
            mu_values: 1D array of avgmu values for signal events.
            discriminant_values: 1D array of raw model outputs for signal events.
            target_pd: Target signal detection probability (e.g. 0.80, 0.90).
            working_point_name: Optional descriptive label for the operating point.

        Returns:
            OperatingPointCalibration containing slice points, fit parameters, and metrics.
        """
        if not (0.0 < target_pd < 1.0):
            raise ValueError(f"target_pd must be in range (0, 1), received {target_pd}.")

        valid_mask = np.isfinite(mu_values) & np.isfinite(discriminant_values)
        mu_clean = mu_values[valid_mask]
        disc_clean = discriminant_values[valid_mask]

        if len(mu_clean) < 10:
            raise ValueError("Insufficient signal events for pileup threshold calibration.")

        if isinstance(self.bins_mu, int):
            if self.mu_range is not None:
                low, high = float(self.mu_range[0]), float(self.mu_range[1])
            else:
                low = float(np.min(mu_clean))
                high = float(np.max(mu_clean))
            if low == high:
                low -= 1.0
                high += 1.0
            bin_edges = np.linspace(low, high, self.bins_mu + 1)
        else:
            bin_edges = np.asarray(self.bins_mu, dtype=float)

        n_bins = len(bin_edges) - 1
        centers_list: List[float] = []
        thresh_list: List[float] = []
        err_list: List[float] = []

        quantile_target = 1.0 - target_pd
        rng = np.random.default_rng(self.random_seed)

        for i in range(n_bins):
            in_slice = (mu_clean >= bin_edges[i]) & (
                (mu_clean < bin_edges[i + 1]) if i < n_bins - 1 else (mu_clean <= bin_edges[i + 1])
            )
            disc_slice = disc_clean[in_slice]
            n_events = len(disc_slice)

            if n_events < 5:
                continue

            center_val = float(0.5 * (bin_edges[i] + bin_edges[i + 1]))
            thresh_val = float(np.quantile(disc_slice, quantile_target))

            if self.n_bootstraps > 1 and n_events > 1:
                boot_samples = rng.choice(disc_slice, size=(self.n_bootstraps, n_events), replace=True)
                boot_quantiles = np.quantile(boot_samples, quantile_target, axis=1)
                thresh_err = float(np.std(boot_quantiles, ddof=1))
            else:
                thresh_err = 0.0

            centers_list.append(center_val)
            thresh_list.append(thresh_val)
            err_list.append(thresh_err)

        if len(centers_list) < 2:
            raise RuntimeError("Fewer than 2 mu slices had sufficient events for linear fitting.")

        centers_arr = np.asarray(centers_list, dtype=float)
        thresh_arr = np.asarray(thresh_list, dtype=float)
        err_arr = np.asarray(err_list, dtype=float)

        reg = stats.linregress(centers_arr, thresh_arr)

        label = working_point_name or f"PD_{int(round(target_pd * 100))}"

        return OperatingPointCalibration(
            working_point_name=label,
            target_pd=target_pd,
            mu_bin_centers=centers_arr,
            slice_thresholds=thresh_arr,
            slice_threshold_errors=err_arr,
            slope=float(reg.slope),
            intercept=float(reg.intercept),
            r_squared=float(reg.rvalue**2),
            p_value=float(reg.pvalue),
            std_err=float(reg.stderr),
        )

    def calibrate_multiple_points(
        self,
        mu_values: np.ndarray,
        discriminant_values: np.ndarray,
        target_pds: Dict[str, float],
    ) -> Dict[str, OperatingPointCalibration]:
        """Calibrates threshold functions for multiple operating points simultaneously.

        Args:
            mu_values: 1D array of avgmu values for signal events.
            discriminant_values: 1D array of raw model outputs for signal events.
            target_pds: Mapping of working point name to target detection efficiency.

        Returns:
            Dictionary of working point names to OperatingPointCalibration objects.
        """
        results: Dict[str, OperatingPointCalibration] = {}
        for wp_name, pd_val in target_pds.items():
            results[wp_name] = self.calibrate_working_point(
                mu_values=mu_values,
                discriminant_values=discriminant_values,
                target_pd=pd_val,
                working_point_name=wp_name,
            )
        return results

    @staticmethod
    def apply_threshold(
        discriminant_values: np.ndarray,
        mu_values: np.ndarray,
        slope: float,
        intercept: float,
    ) -> np.ndarray:
        """Applies linear threshold decision boundary T(mu) = slope * mu + intercept.

        Args:
            discriminant_values: 1D array of model outputs.
            mu_values: 1D array of avgmu values.
            slope: Slope coefficient a.
            intercept: Intercept coefficient b.

        Returns:
            1D array of binary predictions (1 if accepted, 0 otherwise).
        """
        thresholds = slope * mu_values + intercept
        return (discriminant_values >= thresholds).astype(int)

