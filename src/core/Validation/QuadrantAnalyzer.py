import gc
import logging
import os
import pathlib
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy import stats
import torch
import yaml

from src.core.Datasets.SplitManifest import SplitManifest
from src.core.Interfaces.BaseEvaluator import BaseEvaluator
from src.core.Interfaces.BaseResultAggregator import BaseResultAggregator
from src.core.Interfaces.BaseTrainer import BaseTrainer
from src.core.Trainers.NeuralRingerTrainer import NeuralRingerTrainer
from src.core.Validation.FastPhotonCutEvaluator import (
    FastPhotonCutEvaluator,
    UserKinematicGrid,
)
from src.core.Validation.HoldoutEvaluator import HoldoutEvaluator
from src.core.Validation.ResultAggregator import ResultAggregator
from src.Parser.NeuralRingerTrainerConfiguration import NeuralRingerTrainerConfiguration
from src.utils import get_et_eta

log = logging.getLogger(__name__)


@dataclass
class StrategyMetrics:
    """Encapsulates single strategy classification performance metrics.

    Attributes:
        pd: Probability of detection / true positive rate (TPR).
        pf: Probability of false alarm / false positive rate (FPR).
        sp: ATLAS standard SP metric.
        eff: Overall efficiency (accuracy).
        tp: True positives count.
        fp: False positives count.
        tn: True negatives count.
        fn: False negatives count.
        total: Total event count.
    """

    pd: float
    pf: float
    sp: float
    eff: float
    tp: int
    fp: int
    tn: int
    fn: int
    total: int

    @classmethod
    def from_predictions(cls, preds: np.ndarray, labels: np.ndarray) -> "StrategyMetrics":
        tp = int(np.sum((preds == 1) & (labels == 1)))
        fn = int(np.sum((preds == 0) & (labels == 1)))
        tn = int(np.sum((preds == 0) & (labels == 0)))
        fp = int(np.sum((preds == 1) & (labels == 0)))
        total = len(labels)

        n_sig = tp + fn
        n_bg = tn + fp

        pd = float(tp / n_sig) if n_sig > 0 else 0.0
        pf = float(fp / n_bg) if n_bg > 0 else 0.0
        eff = float((tp + tn) / total) if total > 0 else 0.0

        sp = (
            float(np.sqrt(np.sqrt(pd * (1.0 - pf)) * (0.5 * (pd + (1.0 - pf)))))
            if (pd >= 0.0 and pf <= 1.0)
            else 0.0
        )

        return cls(
            pd=pd,
            pf=pf,
            sp=sp,
            eff=eff,
            tp=tp,
            fp=fp,
            tn=tn,
            fn=fn,
            total=total,
        )

    @property
    def bg_eff(self) -> float:
        n_bg = self.tn + self.fp
        return float(self.tn / n_bg) if n_bg > 0 else 0.0

    @property
    def sig_eff_uncertainty(self) -> float:
        n_sig = self.tp + self.fn
        return float(np.sqrt((self.pd * (1.0 - self.pd)) / n_sig)) if n_sig > 0 else 0.0

    @property
    def bg_eff_uncertainty(self) -> float:
        n_bg = self.tn + self.fp
        p = self.bg_eff
        return float(np.sqrt((p * (1.0 - p)) / n_bg)) if n_bg > 0 else 0.0


@dataclass
class QuadrantMetrics:
    """Encapsulates event counts, percentages, and statistical agreement for four performance quadrants.

    Attributes:
        both_correct: Number of events where both strategies predicted correctly.
        model1_only_correct: Number of events where only Strategy 1 predicted correctly.
        model2_only_correct: Number of events where only Strategy 2 predicted correctly.
        both_wrong: Number of events where both strategies predicted incorrectly.
        total_events: Total number of evaluated events.
    """

    both_correct: int
    model1_only_correct: int
    model2_only_correct: int
    both_wrong: int
    total_events: int

    @property
    def strategy1_only_correct(self) -> int:
        return self.model1_only_correct

    @property
    def strategy2_only_correct(self) -> int:
        return self.model2_only_correct

    @property
    def both_correct_ratio(self) -> float:
        return float(self.both_correct / self.total_events) if self.total_events > 0 else 0.0

    @property
    def model1_only_correct_ratio(self) -> float:
        return float(self.model1_only_correct / self.total_events) if self.total_events > 0 else 0.0

    @property
    def strategy1_only_correct_ratio(self) -> float:
        return self.model1_only_correct_ratio

    @property
    def model2_only_correct_ratio(self) -> float:
        return float(self.model2_only_correct / self.total_events) if self.total_events > 0 else 0.0

    @property
    def strategy2_only_correct_ratio(self) -> float:
        return self.model2_only_correct_ratio

    @property
    def both_wrong_ratio(self) -> float:
        return float(self.both_wrong / self.total_events) if self.total_events > 0 else 0.0

    @property
    def both_correct_uncertainty(self) -> float:
        if self.total_events <= 0:
            return 0.0
        p = self.both_correct_ratio
        return float(np.sqrt((p * (1.0 - p)) / self.total_events))

    @property
    def model1_only_correct_uncertainty(self) -> float:
        if self.total_events <= 0:
            return 0.0
        p = self.model1_only_correct_ratio
        return float(np.sqrt((p * (1.0 - p)) / self.total_events))

    @property
    def strategy1_only_correct_uncertainty(self) -> float:
        return self.model1_only_correct_uncertainty

    @property
    def model2_only_correct_uncertainty(self) -> float:
        if self.total_events <= 0:
            return 0.0
        p = self.model2_only_correct_ratio
        return float(np.sqrt((p * (1.0 - p)) / self.total_events))

    @property
    def strategy2_only_correct_uncertainty(self) -> float:
        return self.model2_only_correct_uncertainty

    @property
    def both_wrong_uncertainty(self) -> float:
        if self.total_events <= 0:
            return 0.0
        p = self.both_wrong_ratio
        return float(np.sqrt((p * (1.0 - p)) / self.total_events))

    @property
    def mcnemar_statistic(self) -> float:
        b = self.model1_only_correct
        c = self.model2_only_correct
        if b + c == 0:
            return 0.0
        return float((abs(b - c) - 1.0) ** 2 / (b + c))

    @property
    def mcnemar_p_value(self) -> float:
        b = self.model1_only_correct
        c = self.model2_only_correct
        if b + c == 0:
            return 1.0
        if b + c < 25:
            res = stats.binomtest(min(b, c), b + c, 0.5, alternative="two-sided")
            return float(res.pvalue)
        return float(stats.chi2.sf(self.mcnemar_statistic, df=1))

    def to_matrix(self) -> np.ndarray:
        return np.array(
            [
                [self.both_correct, self.model1_only_correct],
                [self.model2_only_correct, self.both_wrong],
            ],
            dtype=int,
        )

    def to_ratio_matrix(self) -> np.ndarray:
        return np.array(
            [
                [self.both_correct_ratio, self.model1_only_correct_ratio],
                [self.model2_only_correct_ratio, self.both_wrong_ratio],
            ],
            dtype=float,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "both_correct": self.both_correct,
            "strategy1_only_correct": self.model1_only_correct,
            "strategy2_only_correct": self.model2_only_correct,
            "both_wrong": self.both_wrong,
            "total_events": self.total_events,
            "both_correct_ratio": self.both_correct_ratio,
            "both_correct_uncertainty": self.both_correct_uncertainty,
            "strategy1_only_correct_ratio": self.model1_only_correct_ratio,
            "strategy1_only_correct_uncertainty": self.model1_only_correct_uncertainty,
            "strategy2_only_correct_ratio": self.model2_only_correct_ratio,
            "strategy2_only_correct_uncertainty": self.model2_only_correct_uncertainty,
            "both_wrong_ratio": self.both_wrong_ratio,
            "both_wrong_uncertainty": self.both_wrong_uncertainty,
            "mcnemar_statistic": self.mcnemar_statistic,
            "mcnemar_p_value": self.mcnemar_p_value,
        }


@dataclass
class RegionalQuadrantResult:
    """Encapsulates regional quadrant metrics and shower shape distributions for two strategies.

    Attributes:
        iet: Transverse energy bin index.
        ieta: Pseudorapidity bin index.
        strategy1_name: Identifier name for Strategy 1.
        strategy2_name: Identifier name for Strategy 2.
        metrics_strategy1: StrategyMetrics for Strategy 1.
        metrics_strategy2: StrategyMetrics for Strategy 2.
        overall_metrics: QuadrantMetrics for all holdout events.
        signal_metrics: QuadrantMetrics for signal events (class 1).
        background_metrics: QuadrantMetrics for background events (class 0).
        preds_strategy1: Predictions from Strategy 1.
        preds_strategy2: Predictions from Strategy 2.
        labels: True class labels.
        threshold_strategy1: Decision threshold applied for Strategy 1.
        threshold_strategy2: Decision threshold applied for Strategy 2.
        showershapes: Dictionary of L2 trigger shower shape arrays.
        probs_model1: Optional continuous model 1 probability scores.
        probs_model2: Optional continuous model 2 probability scores.
        et: Optional transverse energy values in GeV.
        eta: Optional pseudorapidity values.
    """

    iet: int
    ieta: int
    strategy1_name: str
    strategy2_name: str
    metrics_strategy1: StrategyMetrics
    metrics_strategy2: StrategyMetrics
    overall_metrics: QuadrantMetrics
    signal_metrics: QuadrantMetrics
    background_metrics: QuadrantMetrics
    preds_strategy1: np.ndarray
    preds_strategy2: np.ndarray
    labels: np.ndarray
    threshold_strategy1: float
    threshold_strategy2: float
    showershapes: Dict[str, np.ndarray]
    probs_model1: Optional[np.ndarray] = None
    probs_model2: Optional[np.ndarray] = None
    et: Optional[np.ndarray] = None
    eta: Optional[np.ndarray] = None

    @property
    def model1_name(self) -> str:
        return self.strategy1_name

    @property
    def model2_name(self) -> str:
        return self.strategy2_name

    @property
    def threshold_model1(self) -> float:
        return self.threshold_strategy1

    @property
    def threshold_model2(self) -> float:
        return self.threshold_strategy2


class QuadrantAnalyzer:
    """Orchestrates quadrant analysis comparing models or cut-based selection across kinematic regions.

    Attributes:
        mode: Comparison mode ('model_vs_model' or 'model_vs_cut').
        config1: Parsed configuration instance for Strategy 1 (Model 1).
        data_path1: Results directory holding Model 1 runs.
        working_point: Operating point when running cut-based comparison.
    """

    def __init__(
        self,
        config_path1: str | pathlib.Path,
        data_path1: str | pathlib.Path,
        config_path2: str | pathlib.Path | None = None,
        data_path2: str | pathlib.Path | None = None,
        manifest_path: str | pathlib.Path | None = None,
        threshold_mode: str = "calibrated",
        efficiencies_csv: str | pathlib.Path | None = None,
        default_target_pd: float = 0.9424,
        mode: str = "model_vs_model",
        working_point: str = "loose",
    ) -> None:
        """Initializes QuadrantAnalyzer with configurations and strategy options.

        Args:
            config_path1: Path to Model 1 configuration YAML.
            data_path1: Directory containing Model 1 results and split manifest.
            config_path2: Optional path to Model 2 configuration YAML (for model_vs_model).
            data_path2: Optional directory containing Model 2 results (for model_vs_model).
            manifest_path: Optional path to split manifest JSON file.
            threshold_mode: 'calibrated' for target PD cut, 'default' for 0.5 threshold.
            efficiencies_csv: Optional path to reference operating point table.
            default_target_pd: Target signal detection efficiency.
            mode: Comparison mode ('model_vs_model' or 'model_vs_cut').
            working_point: ATLAS cut working point ('loose', 'medium', 'tight') for model_vs_cut.
        """
        self.mode: str = mode
        self.working_point: str = working_point
        self.threshold_mode: str = threshold_mode
        self.default_target_pd: float = default_target_pd

        self.config1: NeuralRingerTrainerConfiguration = self._load_config(config_path1)
        self.data_path1: pathlib.Path = pathlib.Path(data_path1)

        manifest_file = (
            pathlib.Path(manifest_path)
            if manifest_path is not None
            else self.data_path1 / "split_manifest.json"
        )
        self.manifest: SplitManifest = SplitManifest(str(manifest_file))
        self.reference_eff_df: Optional[pd.DataFrame] = self._load_reference_efficiencies(efficiencies_csv)

        self.trainer1: BaseTrainer = NeuralRingerTrainer(self.config1)
        self.model1 = self.trainer1.factory.create_model()
        self.aggregator1: BaseResultAggregator = ResultAggregator(self.data_path1)
        self.evaluator: BaseEvaluator = HoldoutEvaluator(default_target_pd=default_target_pd)

        self.grid: UserKinematicGrid = UserKinematicGrid()
        self.cut_evaluator: FastPhotonCutEvaluator = FastPhotonCutEvaluator(grid=self.grid)

        if self.mode == "model_vs_model":
            config2_path = config_path2 if config_path2 is not None else config_path1
            if data_path2 is None:
                raise ValueError("data_path2 is required when mode is 'model_vs_model'.")
            self.config2: Optional[NeuralRingerTrainerConfiguration] = self._load_config(config2_path)
            self.data_path2: Optional[pathlib.Path] = pathlib.Path(data_path2)
            self.trainer2: Optional[BaseTrainer] = NeuralRingerTrainer(self.config2)
            self.model2 = self.trainer2.factory.create_model()
            self.aggregator2: Optional[BaseResultAggregator] = ResultAggregator(self.data_path2)
        else:
            self.config2 = None
            self.data_path2 = None
            self.trainer2 = None
            self.model2 = None
            self.aggregator2 = None

    @staticmethod
    def _load_config(config_path: str | pathlib.Path) -> NeuralRingerTrainerConfiguration:
        with open(config_path, "r") as file:
            yaml_data = yaml.safe_load(file)
            return NeuralRingerTrainerConfiguration.model_validate(yaml_data)

    @staticmethod
    def _load_reference_efficiencies(path: str | pathlib.Path | None) -> Optional[pd.DataFrame]:
        if path is not None and os.path.exists(path):
            log.info(f"Loading reference efficiencies from {path}")
            return pd.read_csv(path)
        return None

    def _get_target_pd(self, iet: int, ieta: int) -> float:
        if self.reference_eff_df is not None:
            ref_row = self.reference_eff_df[
                (self.reference_eff_df["et_bin"] == iet)
                & (self.reference_eff_df["eta_bin"] == ieta)
            ]
            if not ref_row.empty:
                return float(ref_row["eff_sig_loose"].values[0])
        return self.default_target_pd

    @staticmethod
    def _get_model_ring_tag(config: NeuralRingerTrainerConfiguration) -> str:
        model_name = config.model.object_name or "Model"
        rings = None
        if config.model.parameters and "input_dim" in config.model.parameters:
            rings = config.model.parameters["input_dim"]
        if rings is None:
            match = re.search(r"(\d+)\s*Rings", config.config_name, re.IGNORECASE)
            if match:
                rings = match.group(1)
        if rings is not None:
            return f"{model_name}_{rings}rings"
        return model_name

    @property
    def comparison_tag(self) -> str:
        """Constructs a descriptive tag representing the strategy comparison.

        Returns:
            Comparison string formatted as <model1>_vs_<model2> or <model1>_vs_Cut_<wp>.
        """
        tag1 = self._get_model_ring_tag(self.config1)
        if self.mode == "model_vs_cut":
            return f"{tag1}_vs_Cut_{self.working_point.capitalize()}"
        tag2 = self._get_model_ring_tag(self.config2) if self.config2 else "Model2"
        return f"{tag1}_vs_{tag2}"

    def get_output_dir(self, base_dir: str | pathlib.Path = "Plots/quandrantic_analysis") -> pathlib.Path:
        """Resolves target output directory by appending the comparison subfolder name.

        Args:
            base_dir: Destination or root directory for quadrant analysis plots.

        Returns:
            Resolved pathlib.Path ending with the comparison tag.
        """
        base_path = pathlib.Path(base_dir)
        if base_path.name == self.comparison_tag:
            return base_path
        return base_path / self.comparison_tag

    @staticmethod
    def compute_quadrant_metrics(
        preds_strategy1: np.ndarray, preds_strategy2: np.ndarray, labels: np.ndarray
    ) -> QuadrantMetrics:
        """Computes quadrant counts and proportions comparing two prediction sets.

        Args:
            preds_strategy1: Binary predictions from Strategy 1.
            preds_strategy2: Binary predictions from Strategy 2.
            labels: Ground truth binary labels.

        Returns:
            QuadrantMetrics instance.
        """
        correct1 = preds_strategy1 == labels
        correct2 = preds_strategy2 == labels

        both_correct = int(np.sum(correct1 & correct2))
        s1_only = int(np.sum(correct1 & ~correct2))
        s2_only = int(np.sum(~correct1 & correct2))
        both_wrong = int(np.sum(~correct1 & ~correct2))
        total_events = len(labels)

        return QuadrantMetrics(
            both_correct=both_correct,
            model1_only_correct=s1_only,
            model2_only_correct=s2_only,
            both_wrong=both_wrong,
            total_events=total_events,
        )

    @staticmethod
    def _build_region_index_map(trainer: BaseTrainer) -> Dict[Tuple[int, int], int]:
        mapping: Dict[Tuple[int, int], int] = {}
        for idx, path in enumerate(trainer.full_dataset.file_paths):
            try:
                iet, ieta = get_et_eta(path)
                mapping[(iet, ieta)] = idx
            except ValueError:
                continue
        return mapping

    @property
    def region_index_map1(self) -> Dict[Tuple[int, int], int]:
        return self._build_region_index_map(self.trainer1)

    @property
    def region_index_map2(self) -> Dict[Tuple[int, int], int]:
        if self.trainer2 is None:
            return {}
        return self._build_region_index_map(self.trainer2)

    @staticmethod
    def extract_l2_showershapes(
        features_full: np.ndarray, feature_names: List[str]
    ) -> Dict[str, np.ndarray]:
        """Extracts and computes L2 calorimeter shower shape distributions.

        Args:
            features_full: Full 2D feature matrix of holdout events.
            feature_names: List or array of column names matching features_full.

        Returns:
            Dictionary mapping shower shape variable names to 1D NumPy arrays.
        """
        name_to_idx = {str(name).strip(): idx for idx, name in enumerate(feature_names)}
        shapes: Dict[str, np.ndarray] = {}

        raw_l2_vars = [
            "trig_L2_calo_weta2",
            "trig_L2_calo_wstot",
            "trig_L2_calo_fracs1",
            "trig_L2_calo_ehad1",
            "trig_L2_calo_emaxs1",
            "trig_L2_calo_e2tsts1",
            "trig_L2_calo_e237",
            "trig_L2_calo_e277",
        ]
        for var in raw_l2_vars:
            if var in name_to_idx:
                shapes[var] = features_full[:, name_to_idx[var]].copy()

        if "trig_L2_calo_e237" in name_to_idx and "trig_L2_calo_e277" in name_to_idx:
            e237 = features_full[:, name_to_idx["trig_L2_calo_e237"]]
            e277 = features_full[:, name_to_idx["trig_L2_calo_e277"]]
            safe_e277 = np.where(e277 != 0.0, e277, 1.0)
            shapes["Rcore"] = np.where(e277 != 0.0, e237 / safe_e277, 0.0)

        if "trig_L2_calo_ehad1" in name_to_idx and "trig_L2_calo_et" in name_to_idx:
            ehad1 = features_full[:, name_to_idx["trig_L2_calo_ehad1"]]
            et = features_full[:, name_to_idx["trig_L2_calo_et"]]
            safe_et = np.where(et != 0.0, et, 1.0)
            shapes["Rhad"] = np.where(et != 0.0, ehad1 / safe_et, 0.0)

        if "trig_L2_calo_emaxs1" in name_to_idx and "trig_L2_calo_e2tsts1" in name_to_idx:
            emaxs1 = features_full[:, name_to_idx["trig_L2_calo_emaxs1"]]
            e2tsts1 = features_full[:, name_to_idx["trig_L2_calo_e2tsts1"]]
            denom = emaxs1 + e2tsts1
            safe_denom = np.where(denom != 0.0, denom, 1.0)
            shapes["Eratio"] = np.where(denom != 0.0, (emaxs1 - e2tsts1) / safe_denom, 0.0)

        return shapes

    def analyze_region(
        self, region: Union[int, Tuple[int, int]]
    ) -> Optional[RegionalQuadrantResult]:
        """Analyzes holdout test events for a specific kinematic region.

        Args:
            region: Dataset file index in Model 1 dataset or (iet, ieta) tuple.

        Returns:
            RegionalQuadrantResult or None if region results cannot be located.
        """
        if isinstance(region, tuple):
            iet, ieta = region
            idx1 = self.region_index_map1.get((iet, ieta))
            if idx1 is None:
                log.warning(f"[iet{iet}.ieta{ieta}] Skipped: region not found in Model 1 dataset.")
                return None
        else:
            idx1 = region
            data_path = self.trainer1.full_dataset.file_paths[idx1]
            iet, ieta = get_et_eta(data_path)

        region_tag = f"iet{iet}.ieta{ieta}"
        region_key = f"et_{iet}_eta_{ieta}"

        if not self.aggregator1.has_region(iet, ieta):
            log.info(f"[{region_tag}] Skipped: Model 1 has no trained results in {self.data_path1}.")
            return None

        if self.mode == "model_vs_model":
            if self.aggregator2 is None or not self.aggregator2.has_region(iet, ieta):
                log.info(f"[{region_tag}] Skipped: Model 2 has no trained results in {self.data_path2}.")
                return None
            idx2 = self.region_index_map2.get((iet, ieta))
            if idx2 is None:
                log.warning(f"[{region_tag}] Skipped: region not found in Model 2 dataset.")
                return None

        if region_key not in self.manifest.data:
            log.warning(f"[{region_tag}] Skipped: region key '{region_key}' not found in split manifest.")
            return None

        test_indices, _ = self.manifest.get_region_splits(region_key)
        if len(test_indices) == 0:
            log.warning(f"[{region_tag}] Skipped: no holdout test events found in manifest.")
            return None

        target_pd = self._get_target_pd(iet, ieta)

        try:
            best_model1, best_rep1, mean_sp1, std_sp1 = self.aggregator1.get_best_model_for_region(iet, ieta)
            self.model1.load_state_dict(best_model1["best_weights"])
        except Exception as e:
            log.warning(f"[{region_tag}] Skipped: Failed loading best model for Model 1: {e}")
            return None

        data1, target1, _ = self.trainer1.full_dataset[idx1]
        ring_indices1 = self.trainer1.full_dataset.ring_column_indices
        if ring_indices1 is None:
            ring_indices1 = getattr(self.trainer1.full_dataset, "indexes", None)

        res1 = self.evaluator.evaluate(
            model=self.model1,
            data=data1,
            target=target1,
            test_indices=test_indices,
            ring_column_indices=ring_indices1,
            device=self.trainer1.device,
            target_pd=target_pd,
        )

        probs1 = res1.predicted_probabilities
        if self.threshold_mode == "calibrated":
            preds1 = res1.predictions_calibrated_cut
            th1 = res1.calibrated_decision_threshold
        else:
            preds1 = res1.predictions_default_threshold
            th1 = 0.5
        y = res1.holdout_labels
        holdout_features = res1.holdout_features_full

        feature_names = getattr(self.trainer1.full_dataset, "feature_names", None)
        if feature_names is not None and isinstance(feature_names, (list, np.ndarray)) and len(feature_names) > 0:
            feature_names_list = [str(f) for f in feature_names]
            eta_matches = np.where(np.array(feature_names_list) == "trig_L2_calo_eta")[0]
            et_matches = np.where(np.array(feature_names_list) == "trig_L2_calo_et")[0]
            eta_idx = int(eta_matches[0]) if len(eta_matches) > 0 else None
            et_idx = int(et_matches[0]) if len(et_matches) > 0 else None
        else:
            feature_names_list = []
            eta_idx = None
            et_idx = None

        if eta_idx is not None and et_idx is not None and holdout_features.shape[1] > max(eta_idx, et_idx):
            eta_vals = holdout_features[:, eta_idx].copy()
            et_vals = (holdout_features[:, et_idx] / 1000.0).copy()
        else:
            eta_vals = np.zeros(len(y), dtype=float)
            et_vals = np.zeros(len(y), dtype=float)

        showershapes = self.extract_l2_showershapes(holdout_features, feature_names_list)

        del data1, target1, res1
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if self.mode == "model_vs_cut":
            athena_threshold = self.grid.get_athena_threshold_for_et_bin(iet)
            pass_cut = self.cut_evaluator.evaluate_events_for_threshold(
                data=holdout_features,
                feature_names=feature_names_list,
                threshold=athena_threshold,
                working_point=self.working_point,
            )
            preds2 = pass_cut.astype(int)
            probs2 = None
            th2 = athena_threshold
            strategy2_name = f"Athena Cut ({self.working_point.capitalize()})"
            log.info(
                f"[{region_tag}] Evaluating: {self.config1.config_name} (SP={mean_sp1:.4f}±{std_sp1:.4f}) vs "
                f"{strategy2_name} (Thr={athena_threshold} GeV) on {len(test_indices):,} test events"
            )
        else:
            try:
                best_model2, best_rep2, mean_sp2, std_sp2 = self.aggregator2.get_best_model_for_region(iet, ieta)
                self.model2.load_state_dict(best_model2["best_weights"])
            except Exception as e:
                log.warning(f"[{region_tag}] Skipped: Failed loading best model for Model 2: {e}")
                return None

            data2, target2, _ = self.trainer2.full_dataset[idx2]
            ring_indices2 = self.trainer2.full_dataset.ring_column_indices
            if ring_indices2 is None:
                ring_indices2 = getattr(self.trainer2.full_dataset, "indexes", None)

            res2 = self.evaluator.evaluate(
                model=self.model2,
                data=data2,
                target=target2,
                test_indices=test_indices,
                ring_column_indices=ring_indices2,
                device=self.trainer2.device,
                target_pd=target_pd,
            )

            probs2 = res2.predicted_probabilities
            if self.threshold_mode == "calibrated":
                preds2 = res2.predictions_calibrated_cut
                th2 = res2.calibrated_decision_threshold
            else:
                preds2 = res2.predictions_default_threshold
                th2 = 0.5

            strategy2_name = self.config2.config_name
            log.info(
                f"[{region_tag}] Evaluating: {self.config1.config_name} (Rep {best_rep1}, SP={mean_sp1:.4f}±{std_sp1:.4f}) vs "
                f"{strategy2_name} (Rep {best_rep2}, SP={mean_sp2:.4f}±{std_sp2:.4f}) on {len(test_indices):,} test events"
            )

            del data2, target2, res2
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        metrics1 = StrategyMetrics.from_predictions(preds1, y)
        metrics2 = StrategyMetrics.from_predictions(preds2, y)

        signal_mask = y == 1
        background_mask = y == 0

        overall_metrics = self.compute_quadrant_metrics(preds1, preds2, y)
        signal_metrics = self.compute_quadrant_metrics(
            preds1[signal_mask], preds2[signal_mask], y[signal_mask]
        )
        background_metrics = self.compute_quadrant_metrics(
            preds1[background_mask], preds2[background_mask], y[background_mask]
        )

        log.info(
            f"[{region_tag}] S1 ({self.config1.config_name}): Pd={metrics1.pd:.4f}, Pf={metrics1.pf:.4f}, SP={metrics1.sp:.4f}, Eff={metrics1.eff:.4f} | "
            f"S2 ({strategy2_name}): Pd={metrics2.pd:.4f}, Pf={metrics2.pf:.4f}, SP={metrics2.sp:.4f}, Eff={metrics2.eff:.4f}"
        )
        log.info(
            f"[{region_tag}] Quadrants: Both Correct={overall_metrics.both_correct:,} ({overall_metrics.both_correct_ratio:.1%}) | "
            f"S1 Advantage={overall_metrics.model1_only_correct:,} ({overall_metrics.model1_only_correct_ratio:.1%}) | "
            f"S2 Advantage={overall_metrics.model2_only_correct:,} ({overall_metrics.model2_only_correct_ratio:.1%}) | "
            f"Both Wrong={overall_metrics.both_wrong:,} ({overall_metrics.both_wrong_ratio:.1%}) | "
            f"McNemar p={overall_metrics.mcnemar_p_value:.3e}"
        )

        return RegionalQuadrantResult(
            iet=iet,
            ieta=ieta,
            strategy1_name=self.config1.config_name,
            strategy2_name=strategy2_name,
            metrics_strategy1=metrics1,
            metrics_strategy2=metrics2,
            overall_metrics=overall_metrics,
            signal_metrics=signal_metrics,
            background_metrics=background_metrics,
            preds_strategy1=preds1,
            preds_strategy2=preds2,
            labels=y,
            threshold_strategy1=th1,
            threshold_strategy2=th2,
            showershapes=showershapes,
            probs_model1=probs1,
            probs_model2=probs2,
            et=et_vals,
            eta=eta_vals,
        )

    def run(self) -> List[RegionalQuadrantResult]:
        """Executes quadrant analysis across all configured kinematic regions.

        Returns:
            List of RegionalQuadrantResult objects for all processed regions.

        Raises:
            FileNotFoundError: If the split manifest cannot be found.
        """
        if not self.manifest.exists():
            raise FileNotFoundError(
                f"Split manifest not found at {self.manifest.manifest_file}. Quadrant analysis requires persisted splits."
            )
        self.manifest.load()

        total_dataset_regions = len(self.trainer1.full_dataset)
        log.info(f"Scanning {total_dataset_regions} kinematic regions for comparison in mode '{self.mode}'...")

        regional_results: List[RegionalQuadrantResult] = []
        for index in range(total_dataset_regions):
            data_path = self.trainer1.full_dataset.file_paths[index]
            iet, ieta = get_et_eta(data_path)

            if not self.config1.is_region_allowed(iet, ieta):
                log.info(f"[iet{iet}.ieta{ieta}] Skipped: excluded by region filters.")
                continue

            result = self.analyze_region((iet, ieta))
            if result is not None:
                regional_results.append(result)

        log.info(
            f"Quadrant analysis complete: {len(regional_results)}/{total_dataset_regions} "
            f"regions successfully evaluated."
        )
        return regional_results

    def compute_global_result(
        self, regional_results: List[RegionalQuadrantResult]
    ) -> Optional[RegionalQuadrantResult]:
        """Aggregates multiple regional results into a global composite result.

        Args:
            regional_results: List of regional results.

        Returns:
            Consolidated RegionalQuadrantResult or None if list is empty.
        """
        if not regional_results:
            return None

        all_preds1 = np.concatenate([r.preds_strategy1 for r in regional_results])
        all_preds2 = np.concatenate([r.preds_strategy2 for r in regional_results])
        all_labels = np.concatenate([r.labels for r in regional_results])

        global_metrics1 = StrategyMetrics.from_predictions(all_preds1, all_labels)
        global_metrics2 = StrategyMetrics.from_predictions(all_preds2, all_labels)

        total_both_correct = sum(r.overall_metrics.both_correct for r in regional_results)
        total_m1_only = sum(r.overall_metrics.model1_only_correct for r in regional_results)
        total_m2_only = sum(r.overall_metrics.model2_only_correct for r in regional_results)
        total_both_wrong = sum(r.overall_metrics.both_wrong for r in regional_results)
        total_events = sum(r.overall_metrics.total_events for r in regional_results)

        overall_metrics = QuadrantMetrics(
            both_correct=total_both_correct,
            model1_only_correct=total_m1_only,
            model2_only_correct=total_m2_only,
            both_wrong=total_both_wrong,
            total_events=total_events,
        )

        sig_both_correct = sum(r.signal_metrics.both_correct for r in regional_results)
        sig_m1_only = sum(r.signal_metrics.model1_only_correct for r in regional_results)
        sig_m2_only = sum(r.signal_metrics.model2_only_correct for r in regional_results)
        sig_both_wrong = sum(r.signal_metrics.both_wrong for r in regional_results)
        sig_events = sum(r.signal_metrics.total_events for r in regional_results)

        signal_metrics = QuadrantMetrics(
            both_correct=sig_both_correct,
            model1_only_correct=sig_m1_only,
            model2_only_correct=sig_m2_only,
            both_wrong=sig_both_wrong,
            total_events=sig_events,
        )

        bg_both_correct = sum(r.background_metrics.both_correct for r in regional_results)
        bg_m1_only = sum(r.background_metrics.model1_only_correct for r in regional_results)
        bg_m2_only = sum(r.background_metrics.model2_only_correct for r in regional_results)
        bg_both_wrong = sum(r.background_metrics.both_wrong for r in regional_results)
        bg_events = sum(r.background_metrics.total_events for r in regional_results)

        background_metrics = QuadrantMetrics(
            both_correct=bg_both_correct,
            model1_only_correct=bg_m1_only,
            model2_only_correct=bg_m2_only,
            both_wrong=bg_both_wrong,
            total_events=bg_events,
        )

        mean_th1 = float(np.mean([r.threshold_strategy1 for r in regional_results]))
        mean_th2 = float(np.mean([r.threshold_strategy2 for r in regional_results]))

        all_et = (
            np.concatenate([r.et for r in regional_results if r.et is not None and len(r.et) > 0])
            if any(r.et is not None and len(r.et) > 0 for r in regional_results)
            else np.array([])
        )
        all_eta = (
            np.concatenate([r.eta for r in regional_results if r.eta is not None and len(r.eta) > 0])
            if any(r.eta is not None and len(r.eta) > 0 for r in regional_results)
            else np.array([])
        )

        global_showershapes: Dict[str, np.ndarray] = {}
        first_shapes = regional_results[0].showershapes
        for var in first_shapes.keys():
            matching_arrays = [r.showershapes[var] for r in regional_results if var in r.showershapes]
            if matching_arrays:
                global_showershapes[var] = np.concatenate(matching_arrays)

        probs1_list = [r.probs_model1 for r in regional_results if r.probs_model1 is not None]
        all_probs1 = np.concatenate(probs1_list) if len(probs1_list) == len(regional_results) else None

        probs2_list = [r.probs_model2 for r in regional_results if r.probs_model2 is not None]
        all_probs2 = np.concatenate(probs2_list) if len(probs2_list) == len(regional_results) else None

        return RegionalQuadrantResult(
            iet=-1,
            ieta=-1,
            strategy1_name=regional_results[0].strategy1_name,
            strategy2_name=regional_results[0].strategy2_name,
            metrics_strategy1=global_metrics1,
            metrics_strategy2=global_metrics2,
            overall_metrics=overall_metrics,
            signal_metrics=signal_metrics,
            background_metrics=background_metrics,
            preds_strategy1=all_preds1,
            preds_strategy2=all_preds2,
            labels=all_labels,
            threshold_strategy1=mean_th1,
            threshold_strategy2=mean_th2,
            showershapes=global_showershapes,
            probs_model1=all_probs1,
            probs_model2=all_probs2,
            et=all_et,
            eta=all_eta,
        )

    def build_summary_dataframe(
        self,
        regional_results: List[RegionalQuadrantResult],
        include_global: bool = True,
    ) -> pd.DataFrame:
        """Constructs consolidated summary DataFrame containing per-strategy and quadrant metrics.

        Args:
            regional_results: List of evaluated regional results.
            include_global: Whether to calculate and include composite global results.

        Returns:
            DataFrame containing performance records across regions.
        """
        records: List[Dict[str, Any]] = []
        all_results = list(regional_results)
        if include_global:
            global_res = self.compute_global_result(regional_results)
            if global_res is not None:
                all_results.append(global_res)

        for res in all_results:
            region_str = f"iet{res.iet}.ieta{res.ieta}" if res.iet >= 0 else "GLOBAL"
            s1 = res.metrics_strategy1
            s2 = res.metrics_strategy2
            m = res.overall_metrics
            records.append(
                {
                    "region": region_str,
                    "iet": res.iet,
                    "ieta": res.ieta,
                    "strategy1_name": res.strategy1_name,
                    "strategy2_name": res.strategy2_name,
                    "threshold_strategy1": res.threshold_strategy1,
                    "threshold_strategy2": res.threshold_strategy2,
                    "total_events": m.total_events,
                    "s1_pd": s1.pd,
                    "s1_sig_eff_uncertainty": s1.sig_eff_uncertainty,
                    "s1_bg_eff": s1.bg_eff,
                    "s1_bg_eff_uncertainty": s1.bg_eff_uncertainty,
                    "s1_pf": s1.pf,
                    "s1_sp": s1.sp,
                    "s1_eff": s1.eff,
                    "s2_pd": s2.pd,
                    "s2_sig_eff_uncertainty": s2.sig_eff_uncertainty,
                    "s2_bg_eff": s2.bg_eff,
                    "s2_bg_eff_uncertainty": s2.bg_eff_uncertainty,
                    "s2_pf": s2.pf,
                    "s2_sp": s2.sp,
                    "s2_eff": s2.eff,
                    "both_correct": m.both_correct,
                    "both_correct_ratio": m.both_correct_ratio,
                    "both_correct_uncertainty": m.both_correct_uncertainty,
                    "s1_only_correct": m.model1_only_correct,
                    "s1_only_correct_ratio": m.model1_only_correct_ratio,
                    "s1_only_correct_uncertainty": m.model1_only_correct_uncertainty,
                    "s2_only_correct": m.model2_only_correct,
                    "s2_only_correct_ratio": m.model2_only_correct_ratio,
                    "s2_only_correct_uncertainty": m.model2_only_correct_uncertainty,
                    "both_wrong": m.both_wrong,
                    "both_wrong_ratio": m.both_wrong_ratio,
                    "both_wrong_uncertainty": m.both_wrong_uncertainty,
                    "mcnemar_statistic": m.mcnemar_statistic,
                    "mcnemar_p_value": m.mcnemar_p_value,
                }
            )
        return pd.DataFrame(records)

    def save_results_table(
        self,
        regional_results: List[RegionalQuadrantResult],
        output_dir: str | pathlib.Path,
        filename: str = "quadrant_results.csv",
        include_global: bool = True,
    ) -> pathlib.Path:
        """Saves tabular quadrant analysis results with strategy metrics to CSV.

        Args:
            regional_results: List of evaluated regional results.
            output_dir: Target output directory.
            filename: Output CSV filename (default: 'quadrant_results.csv').
            include_global: Whether to calculate and include composite global results.

        Returns:
            Destination path of saved CSV table.
        """
        df = self.build_summary_dataframe(regional_results, include_global=include_global)
        target_dir = pathlib.Path(output_dir)
        target_dir.mkdir(parents=True, exist_ok=True)
        csv_path = target_dir / filename
        df.to_csv(csv_path, index=False)
        log.info(f"Saved quadrant results table with uncertainties to: {csv_path}")
        return csv_path
