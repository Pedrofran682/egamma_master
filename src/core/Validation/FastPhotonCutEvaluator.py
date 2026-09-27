from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd


class TrigFastPhotonCutMaps:
    """ATLAS Athena TrigFastPhotonCutMaps replicating L2 fast calo photon cut thresholds."""

    ETA_BINS: List[float] = [0.0, 0.6, 0.8, 1.15, 1.37, 1.52, 1.81, 2.01, 2.37, 2.47]

    def __init__(self, threshold: float) -> None:
        """Initializes the cut thresholds for a given ET threshold in GeV.

        Args:
            threshold: Energy threshold in GeV.

        Raises:
            ValueError: If threshold is negative.
        """
        self.threshold: float = float(threshold)
        self.maps_had_et_thr: Dict[str, List[float]] = {}
        self.maps_car_core_thr: Dict[str, List[float]] = {}
        self.maps_cae_ratio_thr: Dict[str, List[float]] = {}
        self._configure_maps()

    def _configure_maps(self) -> None:
        if 0.0 <= self.threshold < 10.0:
            self.maps_had_et_thr = {
                "etcut": [0.1638, 0.1596, 0.1218, 0.1638, 0.0448875, 0.1386, 0.1596, 0.1638, 0.147],
                "loose": [0.1638, 0.1596, 0.1218, 0.1638, 0.0448875, 0.1386, 0.1596, 0.1638, 0.147],
                "medium": [0.0254625, 0.0238875, 0.0270375, 0.0207375, 0.03465, 0.0378, 0.03465, 0.0286125, 0.02625],
                "tight": [0.0254625, 0.0238875, 0.0270375, 0.0207375, 0.03465, 0.0378, 0.03465, 0.0286125, 0.02625],
            }
            self.maps_car_core_thr = {
                "etcut": [0.532, 0.57, 0.646, 0.684, 0.418, 0.722, 0.684, 0.722, 0.76],
                "loose": [0.532, 0.57, 0.646, 0.684, 0.418, 0.722, 0.684, 0.722, 0.76],
                "medium": [0.83125, 0.719625, 0.814625, 0.83125, 0.703, 0.817, 0.83125, 0.8265, 0.719625],
                "tight": [0.83125, 0.719625, 0.814625, 0.83125, 0.703, 0.817, 0.83125, 0.8265, 0.719625],
            }
            self.maps_cae_ratio_thr = {wp: [-999.0] * 9 for wp in ["etcut", "loose", "medium", "tight"]}
        elif 10.0 <= self.threshold < 15.0:
            self.maps_had_et_thr = {
                "etcut": [0.0359625, 0.0343875, 0.0396375, 0.0291375, 0.0501375, 0.0559125, 0.0548625, 0.0538125, 0.0469875],
                "loose": [0.0359625, 0.0343875, 0.0396375, 0.0291375, 0.0501375, 0.0559125, 0.0548625, 0.0538125, 0.0469875],
                "medium": [0.0359625, 0.0343875, 0.0396375, 0.0291375, 0.0501375, 0.0559125, 0.0548625, 0.0538125, 0.0469875],
                "tight": [0.0359625, 0.0343875, 0.0396375, 0.0291375, 0.0501375, 0.0559125, 0.0548625, 0.0538125, 0.0469875],
            }
            self.maps_car_core_thr = {
                "etcut": [0.786125, 0.786125, 0.767125, 0.795625, 0.703, 0.776625, 0.819375, 0.805125, 0.681625],
                "loose": [0.786125, 0.786125, 0.767125, 0.795625, 0.703, 0.776625, 0.819375, 0.805125, 0.681625],
                "medium": [0.786125, 0.786125, 0.767125, 0.795625, 0.703, 0.776625, 0.819375, 0.805125, 0.681625],
                "tight": [0.786125, 0.786125, 0.767125, 0.795625, 0.703, 0.776625, 0.819375, 0.805125, 0.681625],
            }
            self.maps_cae_ratio_thr = {wp: [-999.0] * 9 for wp in ["etcut", "loose", "medium", "tight"]}
        elif 15.0 <= self.threshold < 20.0:
            self.maps_had_et_thr = {
                "etcut": [0.0328125, 0.0312375, 0.0354375, 0.0270375, 0.0459375, 0.0527625, 0.0433125, 0.0485625, 0.0396375],
                "loose": [0.0328125, 0.0312375, 0.0354375, 0.0270375, 0.0459375, 0.0527625, 0.0433125, 0.0485625, 0.0396375],
                "medium": [0.0328125, 0.0312375, 0.0354375, 0.0270375, 0.0459375, 0.0527625, 0.0433125, 0.0485625, 0.0396375],
                "tight": [0.0328125, 0.0312375, 0.0354375, 0.0270375, 0.0459375, 0.0527625, 0.0433125, 0.0485625, 0.0396375],
            }
            self.maps_car_core_thr = {
                "etcut": [0.809875, 0.805125, 0.786125, 0.809875, 0.703, 0.795625, 0.819375, 0.814625, 0.691125],
                "loose": [0.809875, 0.805125, 0.786125, 0.809875, 0.703, 0.795625, 0.819375, 0.814625, 0.691125],
                "medium": [0.809875, 0.805125, 0.786125, 0.809875, 0.703, 0.795625, 0.819375, 0.814625, 0.691125],
                "tight": [0.809875, 0.805125, 0.786125, 0.809875, 0.703, 0.795625, 0.819375, 0.814625, 0.691125],
            }
            self.maps_cae_ratio_thr = {wp: [-999.0] * 9 for wp in ["etcut", "loose", "medium", "tight"]}
        elif self.threshold >= 20.0:
            self.maps_had_et_thr = {
                "etcut": [0.071, 0.062, 0.075, 0.060, 0.051, 0.057, 0.075, 0.072, 0.051],
                "loose": [0.071, 0.062, 0.075, 0.060, 0.051, 0.057, 0.075, 0.072, 0.051],
                "medium": [0.071, 0.062, 0.075, 0.060, 0.051, 0.057, 0.075, 0.072, 0.051],
                "tight": [0.071, 0.062, 0.075, 0.060, 0.051, 0.057, 0.075, 0.072, 0.051],
            }
            self.maps_car_core_thr = {
                "etcut": [0.819375, 0.819375, 0.800375, 0.828875, 0.7125, 0.805125, 0.843125, 0.824125, 0.700625],
                "loose": [0.819375, 0.819375, 0.800375, 0.828875, 0.7125, 0.805125, 0.843125, 0.824125, 0.700625],
                "medium": [0.819375, 0.819375, 0.800375, 0.828875, 0.7125, 0.805125, 0.843125, 0.824125, 0.700625],
                "tight": [0.819375, 0.819375, 0.800375, 0.828875, 0.7125, 0.805125, 0.843125, 0.824125, 0.700625],
            }
            self.maps_cae_ratio_thr = {wp: [-999.0] * 9 for wp in ["etcut", "loose", "medium", "tight"]}
        else:
            raise ValueError(f"Incorrect threshold {self.threshold}: No cuts configured.")

    @classmethod
    def find_eta_bin_index(cls, abs_eta: Union[float, np.ndarray]) -> Union[int, np.ndarray]:
        """Maps absolute pseudorapidity values to the corresponding Athena eta bin index.

        Args:
            abs_eta: Float or numpy array of absolute eta values.

        Returns:
            Integer index (0-8) or array of indices. Values >= 2.47 return -1.
        """
        if isinstance(abs_eta, np.ndarray):
            bins = np.digitize(abs_eta, cls.ETA_BINS) - 1
            bins = np.where((abs_eta < cls.ETA_BINS[0]) | (abs_eta >= cls.ETA_BINS[-1]), -1, bins)
            return bins

        if abs_eta < cls.ETA_BINS[0] or abs_eta >= cls.ETA_BINS[-1]:
            return -1
        for i in range(len(cls.ETA_BINS) - 1):
            if cls.ETA_BINS[i] <= abs_eta < cls.ETA_BINS[i + 1]:
                return i
        return -1


class UserKinematicGrid:
    """Represents the user target ET and eta kinematic partition grid."""

    DEFAULT_ET_EDGES: List[float] = [15.0, 20.0, 30.0, 40.0, 50.0, float("inf")]
    DEFAULT_ETA_EDGES: List[float] = [0.0, 0.8, 1.37, 1.54, 2.37, 2.50, float("inf")]

    def __init__(
        self,
        et_edges: Optional[List[float]] = None,
        eta_edges: Optional[List[float]] = None,
    ) -> None:
        """Initializes the kinematic grid boundaries.

        Args:
            et_edges: Ordered list of ET boundaries in GeV.
            eta_edges: Ordered list of |eta| boundaries.
        """
        self.et_edges: List[float] = et_edges or list(self.DEFAULT_ET_EDGES)
        self.eta_edges: List[float] = eta_edges or list(self.DEFAULT_ETA_EDGES)

    @property
    def num_et_bins(self) -> int:
        return len(self.et_edges) - 1

    @property
    def num_eta_bins(self) -> int:
        return len(self.eta_edges) - 1

    def get_et_bin(self, et_gev: np.ndarray) -> np.ndarray:
        bins = np.digitize(et_gev, self.et_edges) - 1
        return np.where((et_gev < self.et_edges[0]) | (et_gev >= self.et_edges[-1]), -1, bins)

    def get_eta_bin(self, abs_eta: np.ndarray) -> np.ndarray:
        bins = np.digitize(abs_eta, self.eta_edges) - 1
        return np.where((abs_eta < self.eta_edges[0]) | (abs_eta >= self.eta_edges[-1]), -1, bins)

    def get_athena_threshold_for_et_bin(self, et_bin_idx: int) -> float:
        lower_bound = self.et_edges[et_bin_idx]
        return min(lower_bound, 40.0) if lower_bound >= 40.0 else lower_bound


class RegionEfficiencyAccumulator:
    """Streaming accumulator computing cut efficiencies over kinematic (ET, eta) regions."""

    def __init__(self, grid: UserKinematicGrid, working_points: List[str]) -> None:
        """Initializes the accumulator with zero counters.

        Args:
            grid: UserKinematicGrid defining region boundaries.
            working_points: Operating points to track ('loose', 'medium', 'tight', 'etcut').
        """
        self.grid: UserKinematicGrid = grid
        self.working_points: List[str] = working_points
        self.total_counts: np.ndarray = np.zeros(
            (grid.num_et_bins, grid.num_eta_bins), dtype=np.int64
        )
        self.passed_counts: Dict[str, np.ndarray] = {
            wp: np.zeros((grid.num_et_bins, grid.num_eta_bins), dtype=np.int64)
            for wp in working_points
        }

    def update(
        self,
        et_bins: np.ndarray,
        eta_bins: np.ndarray,
        passed_masks: Dict[str, np.ndarray],
    ) -> None:
        """Accumulates event statistics for a single batch of events.

        Args:
            et_bins: 1D array of assigned ET bin indices (-1 for out of bounds).
            eta_bins: 1D array of assigned eta bin indices (-1 for out of bounds).
            passed_masks: Dictionary mapping working point to boolean pass mask.
        """
        valid_region = (et_bins >= 0) & (eta_bins >= 0)
        valid_et = et_bins[valid_region]
        valid_eta = eta_bins[valid_region]

        np.add.at(self.total_counts, (valid_et, valid_eta), 1)

        for wp, passed in passed_masks.items():
            valid_passed = passed[valid_region]
            np.add.at(
                self.passed_counts[wp],
                (valid_et[valid_passed], valid_eta[valid_passed]),
                1,
            )

    def to_dataframe(self) -> pd.DataFrame:
        """Serializes aggregated statistics into a structured pandas DataFrame.

        Returns:
            DataFrame containing et_bin, eta_bin, boundaries, sample totals, and efficiencies.
        """
        rows: List[Dict[str, Any]] = []
        for iet in range(self.grid.num_et_bins):
            et_low = self.grid.et_edges[iet]
            et_high = self.grid.et_edges[iet + 1]
            et_str = f"[{et_low:.0f}, {et_high:.0f})" if et_high != float("inf") else f">={et_low:.0f}"

            for ieta in range(self.grid.num_eta_bins):
                eta_low = self.grid.eta_edges[ieta]
                eta_high = self.grid.eta_edges[ieta + 1]
                eta_str = f"[{eta_low:.2f}, {eta_high:.2f})" if eta_high != float("inf") else f">={eta_low:.2f}"

                total = int(self.total_counts[iet, ieta])
                row: Dict[str, Any] = {
                    "et_bin": iet,
                    "eta_bin": ieta,
                    "et_range": et_str,
                    "eta_range": eta_str,
                    "total_samples": total,
                }
                for wp in self.working_points:
                    passed = int(self.passed_counts[wp][iet, ieta])
                    eff = float(passed / total) if total > 0 else 0.0
                    row[f"passed_{wp}"] = passed
                    row[f"eff_{wp}"] = eff
                    row[f"eff_sig_{wp}"] = eff
                rows.append(row)
        return pd.DataFrame(rows)


class FastPhotonCutEvaluator:
    """Evaluates particle identification efficiencies purely based on Athena trigger cuts."""

    def __init__(
        self,
        grid: Optional[UserKinematicGrid] = None,
    ) -> None:
        """Initializes FastPhotonCutEvaluator.

        Args:
            grid: Optional UserKinematicGrid defining region granularity.
        """
        self.grid: UserKinematicGrid = grid or UserKinematicGrid()
        self._cached_cut_maps: Dict[float, TrigFastPhotonCutMaps] = {}

    def _get_cut_map(self, threshold: float) -> TrigFastPhotonCutMaps:
        if threshold not in self._cached_cut_maps:
            self._cached_cut_maps[threshold] = TrigFastPhotonCutMaps(threshold)
        return self._cached_cut_maps[threshold]

    def evaluate_events_for_threshold(
        self,
        data: np.ndarray,
        feature_names: List[str],
        threshold: float,
        working_point: str = "loose",
    ) -> np.ndarray:
        """Computes boolean pass mask for events given a specific energy threshold.

        Args:
            data: 2D array of event features.
            feature_names: List of column names.
            threshold: Energy threshold in GeV.
            working_point: Working point to evaluate.

        Returns:
            1D boolean array indicating whether each event passed the cut selection.
        """
        feature_dict = {str(name): idx for idx, name in enumerate(feature_names)}
        cut_map = self._get_cut_map(threshold)

        et = data[:, feature_dict["trig_L2_calo_et"]]
        eta = data[:, feature_dict["trig_L2_calo_eta"]]
        e237 = data[:, feature_dict["trig_L2_calo_e237"]]
        e277 = data[:, feature_dict["trig_L2_calo_e277"]]
        ehad1 = data[:, feature_dict["trig_L2_calo_ehad1"]]

        abs_eta = np.abs(eta)
        athena_eta_bins = TrigFastPhotonCutMaps.find_eta_bin_index(abs_eta)
        valid_eta_mask = athena_eta_bins != -1

        safe_e277 = np.where(e277 != 0, e277, 1.0)
        rcore = np.where(e277 != 0, e237 / safe_e277, -999.0)

        safe_et = np.where(et != 0, et, 1.0)
        had_em = np.where(et != 0, ehad1 / safe_et, -999.0)

        car_core_thrs = np.array(cut_map.maps_car_core_thr[working_point])
        had_et_thrs = np.array(cut_map.maps_had_et_thr[working_point])

        clipped_bins = np.clip(athena_eta_bins, 0, len(car_core_thrs) - 1)
        event_rcore_thrs = car_core_thrs[clipped_bins]
        event_hadet_thrs = np.where(et > 90000.0, 999.0, had_et_thrs[clipped_bins])

        pass_rcore = rcore >= event_rcore_thrs
        pass_had_et = had_em <= event_hadet_thrs

        return valid_eta_mask & pass_rcore & pass_had_et

    def process_batch(
        self,
        data: np.ndarray,
        feature_names: List[str],
        accumulator: RegionEfficiencyAccumulator,
    ) -> None:
        """Processes a single batch of events and updates the running accumulator.

        Args:
            data: 2D feature matrix.
            feature_names: List of column names in data.
            accumulator: RegionEfficiencyAccumulator to update.
        """
        feature_dict = {str(name): idx for idx, name in enumerate(feature_names)}
        et = data[:, feature_dict["trig_L2_calo_et"]]
        eta = data[:, feature_dict["trig_L2_calo_eta"]]

        et_gev = et / 1000.0
        abs_eta = np.abs(eta)

        et_bins = self.grid.get_et_bin(et_gev)
        eta_bins = self.grid.get_eta_bin(abs_eta)

        passed_masks: Dict[str, np.ndarray] = {
            wp: np.zeros(data.shape[0], dtype=bool) for wp in accumulator.working_points
        }

        unique_et_bins = np.unique(et_bins[et_bins >= 0])
        for bin_idx in unique_et_bins:
            event_mask_for_bin = et_bins == bin_idx
            threshold = self.grid.get_athena_threshold_for_et_bin(int(bin_idx))

            for wp in accumulator.working_points:
                pass_sub = self.evaluate_events_for_threshold(
                    data[event_mask_for_bin],
                    feature_names,
                    threshold=threshold,
                    working_point=wp,
                )
                passed_masks[wp][event_mask_for_bin] = pass_sub

        accumulator.update(et_bins, eta_bins, passed_masks)

    def evaluate_files(
        self,
        file_paths: List[Union[str, Path]],
        working_points: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Iterates over files sequentially using a memory-efficient accumulator.

        Args:
            file_paths: List of NPZ file paths to evaluate.
            working_points: Operating points to evaluate (defaults to ['loose', 'medium', 'tight']).

        Returns:
            Aggregated DataFrame with efficiencies computed across all (ET, eta) regions.
        """
        if working_points is None:
            working_points = ["loose", "medium", "tight"]

        accumulator = RegionEfficiencyAccumulator(self.grid, working_points)

        for path in file_paths:
            file_path = Path(path)
            with np.load(file_path, allow_pickle=True) as npz:
                data = npz["data"]
                features = [str(f) for f in npz["feature"]]
            self.process_batch(data, features, accumulator)
            del data

        return accumulator.to_dataframe()

    @staticmethod
    def export_reference_csv(df: pd.DataFrame, output_path: Union[str, Path]) -> None:
        """Exports region summary DataFrame to a CSV file compatible with ModelValidator.

        Args:
            df: DataFrame containing region efficiencies.
            output_path: Target CSV file path.
        """
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
