from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SubdetectorDefinition:
    """Defines a calorimeter subdetector ring range and display color."""

    name: str
    start_ring: int
    end_ring: int
    color: str


@dataclass(frozen=True)
class SubdetectorPeakMetrics:
    """Holds peak ring energy fractions for one subdetector and particle class."""

    region: str
    class_name: str
    subdetector: str
    n_rings_subdetector: int
    n_peak_rings: int
    peak_positions: Tuple[int, ...]
    peak_global_rings: Tuple[int, ...]
    ring_1_energy: float
    ring_2_energy: float
    ring_3_energy: float
    peak_energy_sum: float
    tail_energy_sum: float
    subdetector_energy_sum: float
    total_calorimeter_energy: float
    intra_detector_pct: float
    global_calorimeter_pct: float
    subdetector_share_pct: float
    ring_1_intra_pct: float
    ring_2_intra_pct: float
    ring_3_intra_pct: float
    ring_1_global_pct: float
    ring_2_global_pct: float
    ring_3_global_pct: float

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the metrics record to a plain dictionary.

        Returns:
            Dictionary mapping metric names to values.
        """
        return asdict(self)


class RingPeakEnergyAnalyzer:
    """Analyzes energy concentration in the first k rings of each calorimeter subdetector."""

    DEFAULT_SUBDETECTORS: Tuple[SubdetectorDefinition, ...] = (
        SubdetectorDefinition(name="PreSampler", start_ring=0, end_ring=8, color="#1b9e77"),
        SubdetectorDefinition(name="EM1", start_ring=8, end_ring=72, color="#d95f09"),
        SubdetectorDefinition(name="EM2", start_ring=72, end_ring=80, color="#7570b3"),
        SubdetectorDefinition(name="EM3", start_ring=80, end_ring=88, color="#e7298a"),
        SubdetectorDefinition(name="TileCal", start_ring=88, end_ring=100, color="#66a61e"),
    )

    def __init__(
        self,
        top_k: int = 3,
        subdetectors: Optional[Sequence[SubdetectorDefinition]] = None,
    ) -> None:
        """Initializes the peak energy analyzer.

        Args:
            top_k: Number of leading rings per subdetector considered peak rings.
            subdetectors: Optional custom sequence of subdetector definitions.

        Raises:
            ValueError: If `top_k` is less than 1 or `subdetectors` is empty.
        """
        if int(top_k) < 1:
            raise ValueError("top_k must be at least 1.")
        resolved_subdetectors = (
            tuple(subdetectors) if subdetectors is not None else self.DEFAULT_SUBDETECTORS
        )
        if not resolved_subdetectors:
            raise ValueError("subdetectors must contain at least one subdetector definition.")
        self.top_k: int = int(top_k)
        self.subdetectors: Tuple[SubdetectorDefinition, ...] = resolved_subdetectors

    @staticmethod
    def _safe_pct(numerator: float, denominator: float) -> float:
        if abs(denominator) <= 1e-12:
            return 0.0
        return float((numerator / denominator) * 100.0)

    @classmethod
    def resolve_ring_indexes(
        cls,
        n_rings: int,
        ring_indexes: Optional[np.ndarray] = None,
        subdetectors: Optional[Sequence[SubdetectorDefinition]] = None,
    ) -> np.ndarray:
        """Resolves global ring indices for a ring profile vector.

        Args:
            n_rings: Number of active rings in the profile vector.
            ring_indexes: Optional explicit array mapping local positions to global indices.
            subdetectors: Optional sequence of subdetector definitions for proportional mapping.

        Returns:
            1D integer array of global calorimeter ring indices.

        Raises:
            ValueError: If `n_rings <= 0` or explicit `ring_indexes` length mismatches `n_rings`.
        """
        if n_rings <= 0:
            raise ValueError("n_rings must be positive.")
        if ring_indexes is not None:
            resolved = np.asarray(ring_indexes, dtype=int).ravel()
            if len(resolved) != n_rings:
                raise ValueError(
                    f"ring_indexes length ({len(resolved)}) does not match n_rings ({n_rings})."
                )
            return resolved

        active_subdetectors = (
            tuple(subdetectors) if subdetectors is not None else cls.DEFAULT_SUBDETECTORS
        )
        total_span = sum(s.end_ring - s.start_ring for s in active_subdetectors)
        if n_rings == total_span and active_subdetectors == cls.DEFAULT_SUBDETECTORS:
            return np.arange(total_span, dtype=int)

        percentage = float(n_rings) / float(total_span)
        selected: List[int] = []
        for subdet in active_subdetectors:
            full_range = list(range(subdet.start_ring, subdet.end_ring))
            selected.extend(full_range[: round(len(full_range) * percentage)])

        if len(selected) == n_rings:
            return np.asarray(selected, dtype=int)

        return np.arange(n_rings, dtype=int)

    def resolve_subdetector_positions(
        self,
        n_rings: int,
        ring_indexes: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:
        """Maps each subdetector name to 0-based column positions in the profile vector.

        Args:
            n_rings: Number of active rings in the profile vector.
            ring_indexes: Optional explicit array mapping local positions to global indices.

        Returns:
            Dictionary mapping subdetector names to local index arrays.
        """
        resolved = self.resolve_ring_indexes(
            n_rings=n_rings,
            ring_indexes=ring_indexes,
            subdetectors=self.subdetectors,
        )
        return {
            subdet.name: np.where(
                (resolved >= subdet.start_ring) & (resolved < subdet.end_ring)
            )[0]
            for subdet in self.subdetectors
        }

    def describe_subdetector_mapping(
        self,
        n_rings: int,
        ring_indexes: Optional[np.ndarray] = None,
    ) -> pd.DataFrame:
        """Builds a diagnostic table describing ring indices per subdetector.

        Args:
            n_rings: Number of active rings in the model.
            ring_indexes: Optional explicit global ring indices.

        Returns:
            DataFrame summarizing subdetector ring counts, local spans, and peak ring indices.
        """
        resolved = self.resolve_ring_indexes(
            n_rings=n_rings,
            ring_indexes=ring_indexes,
            subdetectors=self.subdetectors,
        )
        positions_map = self.resolve_subdetector_positions(n_rings=n_rings, ring_indexes=resolved)
        rows: List[Dict[str, Any]] = []
        for subdet in self.subdetectors:
            pos = positions_map[subdet.name]
            n_sub = int(len(pos))
            n_peak = min(self.top_k, n_sub)
            peak_pos = pos[:n_peak]
            global_sub = resolved[pos] if n_sub > 0 else np.array([], dtype=int)
            peak_global = resolved[peak_pos] if n_peak > 0 else np.array([], dtype=int)
            rows.append(
                {
                    "subdetector": subdet.name,
                    "n_rings": n_sub,
                    "n_peak_rings": n_peak,
                    "local_positions": f"{int(pos[0])}..{int(pos[-1])}" if n_sub > 0 else "",
                    "global_ring_range": (
                        f"{int(global_sub[0])}..{int(global_sub[-1])}" if n_sub > 0 else ""
                    ),
                    "peak_local_positions": tuple(int(p) for p in peak_pos),
                    "peak_global_rings": tuple(int(g) for g in peak_global),
                }
            )
        return pd.DataFrame(rows)

    def analyze_profile(
        self,
        mean_profile: np.ndarray,
        ring_indexes: Optional[np.ndarray] = None,
        region: str = "global",
        class_name: str = "Signal",
    ) -> List[SubdetectorPeakMetrics]:
        """Computes intra-detector and global calorimeter peak fractions for a single profile.

        Args:
            mean_profile: 1D mean normalized energy profile array across rings.
            ring_indexes: Optional explicit global ring index array.
            region: Kinematic region identifier string.
            class_name: Particle class label (e.g., 'Signal' or 'Background').

        Returns:
            List of SubdetectorPeakMetrics, one per configured subdetector.

        Raises:
            ValueError: If `mean_profile` is empty or not 1D-convertible.
        """
        profile = np.asarray(mean_profile, dtype=np.float64).ravel()
        if profile.size == 0:
            raise ValueError("mean_profile must not be empty.")

        global_indexes = self.resolve_ring_indexes(
            n_rings=int(profile.size),
            ring_indexes=ring_indexes,
            subdetectors=self.subdetectors,
        )
        positions_map = self.resolve_subdetector_positions(
            n_rings=int(profile.size),
            ring_indexes=global_indexes,
        )
        total_energy = float(np.sum(profile))

        metrics: List[SubdetectorPeakMetrics] = []
        for subdet in self.subdetectors:
            pos = positions_map[subdet.name]
            n_sub = int(len(pos))
            n_peak = min(self.top_k, n_sub)
            peak_pos = pos[:n_peak]

            subdet_vals = profile[pos] if n_sub > 0 else np.array([], dtype=np.float64)
            peak_vals = profile[peak_pos] if n_peak > 0 else np.array([], dtype=np.float64)

            subdet_energy = float(np.sum(subdet_vals))
            peak_energy = float(np.sum(peak_vals))
            tail_energy = float(subdet_energy - peak_energy)

            ring_1_energy = float(peak_vals[0]) if n_peak >= 1 else 0.0
            ring_2_energy = float(peak_vals[1]) if n_peak >= 2 else 0.0
            ring_3_energy = float(peak_vals[2]) if n_peak >= 3 else 0.0

            metrics.append(
                SubdetectorPeakMetrics(
                    region=region,
                    class_name=class_name,
                    subdetector=subdet.name,
                    n_rings_subdetector=n_sub,
                    n_peak_rings=n_peak,
                    peak_positions=tuple(int(p) for p in peak_pos),
                    peak_global_rings=tuple(int(global_indexes[p]) for p in peak_pos),
                    ring_1_energy=ring_1_energy,
                    ring_2_energy=ring_2_energy,
                    ring_3_energy=ring_3_energy,
                    peak_energy_sum=peak_energy,
                    tail_energy_sum=tail_energy,
                    subdetector_energy_sum=subdet_energy,
                    total_calorimeter_energy=total_energy,
                    intra_detector_pct=self._safe_pct(peak_energy, subdet_energy),
                    global_calorimeter_pct=self._safe_pct(peak_energy, total_energy),
                    subdetector_share_pct=self._safe_pct(subdet_energy, total_energy),
                    ring_1_intra_pct=self._safe_pct(ring_1_energy, subdet_energy),
                    ring_2_intra_pct=self._safe_pct(ring_2_energy, subdet_energy),
                    ring_3_intra_pct=self._safe_pct(ring_3_energy, subdet_energy),
                    ring_1_global_pct=self._safe_pct(ring_1_energy, total_energy),
                    ring_2_global_pct=self._safe_pct(ring_2_energy, total_energy),
                    ring_3_global_pct=self._safe_pct(ring_3_energy, total_energy),
                )
            )
        return metrics

    def analyze_signal_vs_background(
        self,
        mean_signal: np.ndarray,
        mean_bkg: np.ndarray,
        ring_indexes: Optional[np.ndarray] = None,
        region: str = "global",
        n_rings_label: Optional[int] = None,
    ) -> pd.DataFrame:
        """Builds a side-by-side comparison DataFrame of peak ring shares for Signal vs Background.

        Args:
            mean_signal: Mean normalized ring profile for signal events.
            mean_bkg: Mean normalized ring profile for background events.
            ring_indexes: Optional explicit global ring index array.
            region: Kinematic region identifier string.
            n_rings_label: Optional integer label for model ring dimensionality.

        Returns:
            DataFrame with one row per subdetector comparing Signal and Background metrics.

        Raises:
            ValueError: If `mean_signal` and `mean_bkg` differ in length.
        """
        sig_arr = np.asarray(mean_signal, dtype=np.float64).ravel()
        bkg_arr = np.asarray(mean_bkg, dtype=np.float64).ravel()
        if sig_arr.size != bkg_arr.size:
            raise ValueError(
                f"mean_signal length ({sig_arr.size}) must match mean_bkg length ({bkg_arr.size})."
            )

        sig_metrics = self.analyze_profile(
            mean_profile=sig_arr,
            ring_indexes=ring_indexes,
            region=region,
            class_name="Signal",
        )
        bkg_metrics = self.analyze_profile(
            mean_profile=bkg_arr,
            ring_indexes=ring_indexes,
            region=region,
            class_name="Background",
        )

        resolved_n_rings = int(n_rings_label) if n_rings_label is not None else int(sig_arr.size)
        rows: List[Dict[str, Any]] = []
        for sig, bkg in zip(sig_metrics, bkg_metrics):
            rows.append(
                {
                    "region": region,
                    "n_rings_model": resolved_n_rings,
                    "subdetector": sig.subdetector,
                    "n_rings_subdetector": sig.n_rings_subdetector,
                    "n_peak_rings": sig.n_peak_rings,
                    "peak_global_rings": sig.peak_global_rings,
                    "signal_ring_1_energy": sig.ring_1_energy,
                    "signal_ring_2_energy": sig.ring_2_energy,
                    "signal_ring_3_energy": sig.ring_3_energy,
                    "signal_peak_energy": sig.peak_energy_sum,
                    "signal_subdetector_energy": sig.subdetector_energy_sum,
                    "signal_intra_detector_pct": sig.intra_detector_pct,
                    "signal_global_calorimeter_pct": sig.global_calorimeter_pct,
                    "signal_subdetector_share_pct": sig.subdetector_share_pct,
                    "signal_ring_1_intra_pct": sig.ring_1_intra_pct,
                    "signal_ring_2_intra_pct": sig.ring_2_intra_pct,
                    "signal_ring_3_intra_pct": sig.ring_3_intra_pct,
                    "background_ring_1_energy": bkg.ring_1_energy,
                    "background_ring_2_energy": bkg.ring_2_energy,
                    "background_ring_3_energy": bkg.ring_3_energy,
                    "background_peak_energy": bkg.peak_energy_sum,
                    "background_subdetector_energy": bkg.subdetector_energy_sum,
                    "background_intra_detector_pct": bkg.intra_detector_pct,
                    "background_global_calorimeter_pct": bkg.global_calorimeter_pct,
                    "background_subdetector_share_pct": bkg.subdetector_share_pct,
                    "background_ring_1_intra_pct": bkg.ring_1_intra_pct,
                    "background_ring_2_intra_pct": bkg.ring_2_intra_pct,
                    "background_ring_3_intra_pct": bkg.ring_3_intra_pct,
                    "delta_intra_detector_pct": sig.intra_detector_pct - bkg.intra_detector_pct,
                    "delta_global_calorimeter_pct": (
                        sig.global_calorimeter_pct - bkg.global_calorimeter_pct
                    ),
                }
            )
        return pd.DataFrame(rows)

    def compute_total_peak_concentration(self, comparison_df: pd.DataFrame) -> Dict[str, float]:
        """Summarizes total calorimeter energy captured by all subdetector peak rings combined.

        Args:
            comparison_df: Output DataFrame from `analyze_signal_vs_background`.

        Returns:
            Dictionary with total peak rings count, total ring count, cumulative global %,
            mean intra-detector %, and signal-vs-background delta.
        """
        if comparison_df.empty:
            return {
                "total_peak_rings": 0.0,
                "total_rings": 0.0,
                "signal_total_peak_global_pct": 0.0,
                "background_total_peak_global_pct": 0.0,
                "delta_total_peak_global_pct": 0.0,
                "signal_mean_intra_pct": 0.0,
                "background_mean_intra_pct": 0.0,
            }

        sig_total_global = float(comparison_df["signal_global_calorimeter_pct"].sum())
        bkg_total_global = float(comparison_df["background_global_calorimeter_pct"].sum())
        return {
            "total_peak_rings": float(comparison_df["n_peak_rings"].sum()),
            "total_rings": float(comparison_df["n_rings_subdetector"].sum()),
            "signal_total_peak_global_pct": sig_total_global,
            "background_total_peak_global_pct": bkg_total_global,
            "delta_total_peak_global_pct": sig_total_global - bkg_total_global,
            "signal_mean_intra_pct": float(comparison_df["signal_intra_detector_pct"].mean()),
            "background_mean_intra_pct": float(
                comparison_df["background_intra_detector_pct"].mean()
            ),
        }

    def summarize_across_regions(self, regional_df: pd.DataFrame) -> pd.DataFrame:
        """Aggregates peak percentage distributions across multiple kinematic regions.

        Args:
            regional_df: Concatenated DataFrame of regional `analyze_signal_vs_background` results.

        Returns:
            DataFrame with mean, std, min, and max per subdetector across regions.
        """
        metric_columns = (
            "signal_intra_detector_pct",
            "background_intra_detector_pct",
            "signal_global_calorimeter_pct",
            "background_global_calorimeter_pct",
            "signal_subdetector_share_pct",
            "background_subdetector_share_pct",
        )
        if regional_df.empty:
            columns = ["subdetector", "n_regions"]
            for metric in metric_columns:
                columns.extend([f"{metric}_mean", f"{metric}_std", f"{metric}_min", f"{metric}_max"])
            return pd.DataFrame(columns=columns)

        ordered_names = [s.name for s in self.subdetectors]
        extra_names = [
            name for name in regional_df["subdetector"].unique() if name not in ordered_names
        ]
        summary_rows: List[Dict[str, Any]] = []

        for subdet_name in ordered_names + extra_names:
            subset = regional_df[regional_df["subdetector"] == subdet_name]
            if subset.empty:
                continue
            row: Dict[str, Any] = {
                "subdetector": subdet_name,
                "n_regions": int(subset["region"].nunique()),
            }
            for metric in metric_columns:
                vals = subset[metric].astype(float).to_numpy()
                row[f"{metric}_mean"] = float(np.mean(vals))
                row[f"{metric}_std"] = float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0
                row[f"{metric}_min"] = float(np.min(vals))
                row[f"{metric}_max"] = float(np.max(vals))
            summary_rows.append(row)

        return pd.DataFrame(summary_rows)
