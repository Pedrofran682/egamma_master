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
from src.core.Validation.HoldoutEvaluator import HoldoutEvaluator
from src.core.Validation.ResultAggregator import ResultAggregator
from src.Parser.NeuralRingerTrainerConfiguration import NeuralRingerTrainerConfiguration
from src.utils import get_et_eta

log = logging.getLogger(__name__)


@dataclass
class QuadrantMetrics:
    """Encapsulates event counts, percentages, and statistical agreement for four performance quadrants.

    Attributes:
        both_correct: Number of events where both models predicted correctly.
        model1_only_correct: Number of events where only Model 1 predicted correctly.
        model2_only_correct: Number of events where only Model 2 predicted correctly.
        both_wrong: Number of events where both models predicted incorrectly.
        total_events: Total number of evaluated events.
    """

    both_correct: int
    model1_only_correct: int
    model2_only_correct: int
    both_wrong: int
    total_events: int

    @property
    def both_correct_ratio(self) -> float:
        return float(self.both_correct / self.total_events) if self.total_events > 0 else 0.0

    @property
    def model1_only_correct_ratio(self) -> float:
        return float(self.model1_only_correct / self.total_events) if self.total_events > 0 else 0.0

    @property
    def model2_only_correct_ratio(self) -> float:
        return float(self.model2_only_correct / self.total_events) if self.total_events > 0 else 0.0

    @property
    def both_wrong_ratio(self) -> float:
        return float(self.both_wrong / self.total_events) if self.total_events > 0 else 0.0

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
            "model1_only_correct": self.model1_only_correct,
            "model2_only_correct": self.model2_only_correct,
            "both_wrong": self.both_wrong,
            "total_events": self.total_events,
            "both_correct_ratio": self.both_correct_ratio,
            "model1_only_correct_ratio": self.model1_only_correct_ratio,
            "model2_only_correct_ratio": self.model2_only_correct_ratio,
            "both_wrong_ratio": self.both_wrong_ratio,
            "mcnemar_statistic": self.mcnemar_statistic,
            "mcnemar_p_value": self.mcnemar_p_value,
        }


@dataclass
class RegionalQuadrantResult:
    """Encapsulates regional quadrant metrics and score distributions for two models.

    Attributes:
        iet: Transverse energy bin index.
        ieta: Pseudorapidity bin index.
        model1_name: Identifier name for Model 1.
        model2_name: Identifier name for Model 2.
        overall_metrics: QuadrantMetrics for all holdout events.
        signal_metrics: QuadrantMetrics for signal events (class 1).
        background_metrics: QuadrantMetrics for background events (class 0).
        probs_model1: Model 1 predicted probabilities.
        probs_model2: Model 2 predicted probabilities.
        labels: True class labels.
        threshold_model1: Decision threshold applied for Model 1.
        threshold_model2: Decision threshold applied for Model 2.
    """

    iet: int
    ieta: int
    model1_name: str
    model2_name: str
    overall_metrics: QuadrantMetrics
    signal_metrics: QuadrantMetrics
    background_metrics: QuadrantMetrics
    probs_model1: np.ndarray
    probs_model2: np.ndarray
    labels: np.ndarray
    threshold_model1: float
    threshold_model2: float


class QuadrantAnalyzer:
    """Orchestrates quadrant analysis comparing two models across kinematic regions.

    Attributes:
        config1: Parsed configuration instance for Model 1.
        config2: Parsed configuration instance for Model 2.
        data_path1: Results directory holding Model 1 runs.
        data_path2: Results directory holding Model 2 runs.
        threshold_mode: Decision threshold strategy ('calibrated' or 'default').
        default_target_pd: Fallback target signal efficiency.
    """

    def __init__(
        self,
        config_path1: str | pathlib.Path,
        data_path1: str | pathlib.Path,
        config_path2: str | pathlib.Path,
        data_path2: str | pathlib.Path,
        manifest_path: str | pathlib.Path | None = None,
        threshold_mode: str = "calibrated",
        efficiencies_csv: str | pathlib.Path | None = None,
        default_target_pd: float = 0.9424,
    ) -> None:
        """Initializes QuadrantAnalyzer with configurations and results paths.

        Args:
            config_path1: Path to Model 1 configuration YAML.
            data_path1: Directory containing Model 1 results and split manifest.
            config_path2: Path to Model 2 configuration YAML.
            data_path2: Directory containing Model 2 results.
            manifest_path: Optional path to split manifest JSON file.
            threshold_mode: 'calibrated' for target PD cut, 'default' for 0.5 threshold.
            efficiencies_csv: Optional path to reference operating point table.
            default_target_pd: Target signal detection efficiency.
        """
        self.config1: NeuralRingerTrainerConfiguration = self._load_config(config_path1)
        self.config2: NeuralRingerTrainerConfiguration = self._load_config(config_path2)
        self.data_path1: pathlib.Path = pathlib.Path(data_path1)
        self.data_path2: pathlib.Path = pathlib.Path(data_path2)
        self.threshold_mode: str = threshold_mode
        self.default_target_pd: float = default_target_pd

        manifest_file = (
            pathlib.Path(manifest_path)
            if manifest_path is not None
            else self.data_path1 / "split_manifest.json"
        )
        self.manifest: SplitManifest = SplitManifest(str(manifest_file))
        self.reference_eff_df: Optional[pd.DataFrame] = self._load_reference_efficiencies(efficiencies_csv)

        self.trainer1: BaseTrainer = NeuralRingerTrainer(self.config1)
        self.trainer2: BaseTrainer = NeuralRingerTrainer(self.config2)

        self.model1 = self.trainer1.factory.create_model()
        self.model2 = self.trainer2.factory.create_model()

        self.aggregator1: BaseResultAggregator = ResultAggregator(self.data_path1)
        self.aggregator2: BaseResultAggregator = ResultAggregator(self.data_path2)
        self.evaluator: BaseEvaluator = HoldoutEvaluator(default_target_pd=default_target_pd)

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
        """Constructs a descriptive tag representing the model comparison and ring counts.

        Returns:
            Comparison string formatted as <model1>_<rings1>rings_vs_<model2>_<rings2>rings.
        """
        tag1 = self._get_model_ring_tag(self.config1)
        tag2 = self._get_model_ring_tag(self.config2)
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
        preds_model1: np.ndarray, preds_model2: np.ndarray, labels: np.ndarray
    ) -> QuadrantMetrics:
        """Computes quadrant counts and proportions comparing two prediction sets.

        Args:
            preds_model1: Binary predictions from Model 1.
            preds_model2: Binary predictions from Model 2.
            labels: Ground truth binary labels.

        Returns:
            QuadrantMetrics instance.
        """
        correct_model1 = preds_model1 == labels
        correct_model2 = preds_model2 == labels

        both_correct = int(np.sum(correct_model1 & correct_model2))
        model1_only_correct = int(np.sum(correct_model1 & ~correct_model2))
        model2_only_correct = int(np.sum(~correct_model1 & correct_model2))
        both_wrong = int(np.sum(~correct_model1 & ~correct_model2))
        total_events = len(labels)

        return QuadrantMetrics(
            both_correct=both_correct,
            model1_only_correct=model1_only_correct,
            model2_only_correct=model2_only_correct,
            both_wrong=both_wrong,
            total_events=total_events,
        )

    @staticmethod
    def _build_region_index_map(trainer: NeuralRingerTrainer) -> Dict[Tuple[int, int], int]:
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
        return self._build_region_index_map(self.trainer2)

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

        has_model1 = self.aggregator1.has_region(iet, ieta)
        has_model2 = self.aggregator2.has_region(iet, ieta)

        if not has_model1 and not has_model2:
            log.info(f"[{region_tag}] Skipped: neither Model 1 nor Model 2 have trained results.")
            return None
        if not has_model1:
            log.info(f"[{region_tag}] Skipped: Model 1 has no trained results in {self.data_path1}.")
            return None
        if not has_model2:
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

        try:
            best_model2, best_rep2, mean_sp2, std_sp2 = self.aggregator2.get_best_model_for_region(iet, ieta)
            self.model2.load_state_dict(best_model2["best_weights"])
        except Exception as e:
            log.warning(f"[{region_tag}] Skipped: Failed loading best model for Model 2: {e}")
            return None

        log.info(
            f"[{region_tag}] Evaluating: Model 1 (Rep {best_rep1}, SP={mean_sp1:.4f}±{std_sp1:.4f}) vs "
            f"Model 2 (Rep {best_rep2}, SP={mean_sp2:.4f}±{std_sp2:.4f}) on {len(test_indices):,} test events"
        )

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

        del data1, target1, res1
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

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

        del data2, target2, res2
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

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
            f"[{region_tag}] Results: Both Correct={overall_metrics.both_correct:,} ({overall_metrics.both_correct_ratio:.1%}) | "
            f"M1 Advantage={overall_metrics.model1_only_correct:,} ({overall_metrics.model1_only_correct_ratio:.1%}) | "
            f"M2 Advantage={overall_metrics.model2_only_correct:,} ({overall_metrics.model2_only_correct_ratio:.1%}) | "
            f"Both Wrong={overall_metrics.both_wrong:,} ({overall_metrics.both_wrong_ratio:.1%}) | "
            f"McNemar p={overall_metrics.mcnemar_p_value:.3e}"
        )

        return RegionalQuadrantResult(
            iet=iet,
            ieta=ieta,
            model1_name=self.config1.config_name,
            model2_name=self.config2.config_name,
            overall_metrics=overall_metrics,
            signal_metrics=signal_metrics,
            background_metrics=background_metrics,
            probs_model1=probs1,
            probs_model2=probs2,
            labels=y,
            threshold_model1=th1,
            threshold_model2=th2,
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
        log.info(f"Scanning {total_dataset_regions} kinematic regions for model comparison...")

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
            f"regions successfully evaluated (having models for both Model 1 and Model 2)."
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

        all_probs1 = np.concatenate([r.probs_model1 for r in regional_results])
        all_probs2 = np.concatenate([r.probs_model2 for r in regional_results])
        all_labels = np.concatenate([r.labels for r in regional_results])

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

        mean_th1 = float(np.mean([r.threshold_model1 for r in regional_results]))
        mean_th2 = float(np.mean([r.threshold_model2 for r in regional_results]))

        return RegionalQuadrantResult(
            iet=-1,
            ieta=-1,
            model1_name=regional_results[0].model1_name,
            model2_name=regional_results[0].model2_name,
            overall_metrics=overall_metrics,
            signal_metrics=signal_metrics,
            background_metrics=background_metrics,
            probs_model1=all_probs1,
            probs_model2=all_probs2,
            labels=all_labels,
            threshold_model1=mean_th1,
            threshold_model2=mean_th2,
        )
