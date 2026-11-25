from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import roc_curve

from src.Parser.NeuralRingerTrainerConfiguration import NeuralRingerTrainerConfiguration
from src.core.Datasets.SplitManifest import SplitManifest
from src.core.Plotting.FeatureDistributionPlotter import FeatureDistributionPlotter
from src.core.Plotting.PileupPlotter import PileupPlotter
from src.core.Plotting.QuadrantPlotter import QuadrantPlotter
from src.core.Trainers.NeuralRingerTrainer import NeuralRingerTrainer
from src.core.Validation.DistributionComparator import DistributionComparator
from src.core.Validation.FastPhotonCutEvaluator import FastPhotonCutEvaluator, UserKinematicGrid
from src.core.Validation.HoldoutEvaluator import HoldoutEvaluationResult, HoldoutEvaluator
from src.core.Validation.PileupAnalyzer import PileupAnalyzer
from src.core.Validation.PileupThresholdCalibrator import PileupThresholdCalibrator
from src.core.Validation.QuadrantAnalyzer import (
    QuadrantAnalyzer,
    RegionalQuadrantResult,
    StrategyMetrics,
)
from src.core.Validation.ResultAggregator import ResultAggregator
from src.utils import compute_sp_metric, extract_feature_values, get_et_eta, save_table


VALID_STRATEGY_METHODS = ("best_sp", "hard_cut", "pileup", "athena_cut")


@dataclass(frozen=True)
class ComparisonModelConfig:
    """Identifies a model configuration, results directory, display alias, and optional method."""

    config_path: Optional[Path] = None
    results_path: Optional[Path] = None
    alias: str = ""
    method: Optional[str] = None

    def __post_init__(self) -> None:
        if self.method is not None and self.method not in VALID_STRATEGY_METHODS:
            raise ValueError(
                f"Invalid method '{self.method}'. Expected one of {VALID_STRATEGY_METHODS}."
            )
        if self.method != "athena_cut" and (self.config_path is None or self.results_path is None):
            raise ValueError("`config_path` and `results_path` are required unless method == 'athena_cut'.")

    @classmethod
    def athena_cut(
        cls, alias: Optional[str] = None, working_point: Optional[str] = None
    ) -> "ComparisonModelConfig":
        """Creates a configuration representing the Athena fast photon cut strategy."""
        if alias is None:
            wp_label = f" ({working_point.capitalize()})" if working_point else ""
            alias = f"Athena Cut{wp_label}"
        return cls(config_path=None, results_path=None, alias=alias, method="athena_cut")


@dataclass
class ComparisonPayload:
    """Holds prepared holdout predictions, labels, and metadata for two strategies."""

    mode: str
    region_key: str
    iet: int
    ieta: int
    strategy1_name: str
    strategy2_name: str
    preds_s1: np.ndarray
    preds_s2: np.ndarray
    thresh_s1: float
    thresh_s2: float
    probs_s1: Optional[np.ndarray]
    probs_s2: Optional[np.ndarray]
    test_features: np.ndarray
    test_labels: np.ndarray
    feature_names: np.ndarray
    valid_features: List[str]
    showershapes: Dict[str, np.ndarray]
    et_vals: np.ndarray
    eta_vals: np.ndarray
    preds_athena: Optional[np.ndarray] = None
    athena_name: str = ""
    is_s1_ml: bool = True
    is_s2_ml: bool = True
    mu_test: Optional[np.ndarray] = None
    pileup_calibrations: Optional[List[Dict[str, Any]]] = None


class SingleRegionComparisonRunner:
    """Runs single-region quadrant and feature-distribution comparisons between two strategies."""

    VALID_MODES = (
        "best_sp_vs_hard_cut",
        "best_sp_vs_pileup",
        "hard_cut_vs_pileup",
        "best_sp_vs_best_sp",
        "pileup_vs_pileup",
    )
    MODE_TO_METHODS: Dict[str, Tuple[str, str]] = {
        "best_sp_vs_hard_cut": ("best_sp", "hard_cut"),
        "best_sp_vs_pileup": ("best_sp", "pileup"),
        "hard_cut_vs_pileup": ("hard_cut", "pileup"),
        "best_sp_vs_best_sp": ("best_sp", "best_sp"),
        "pileup_vs_pileup": ("pileup", "pileup"),
    }

    def __init__(
        self,
        iet: int,
        ieta: int,
        model1: ComparisonModelConfig,
        model2: Optional[ComparisonModelConfig] = None,
        compare_same_model: bool = True,
        target_pd_mode: str = "athena",
        custom_target_pd: float = 0.9424,
        working_point: str = "loose",
        data_path: Optional[Path] = None,
        manifest_path: Optional[Path] = None,
        language: str = "pt",
        bins_mu: int = 10,
        n_bootstraps_pileup: int = 100,
        plot_format: str = "png",
        log_y: bool = True,
    ) -> None:
        """Initializes the single-region comparison runner.

        Args:
            iet: Transverse energy bin index.
            ieta: Pseudorapidity bin index.
            model1: Primary model specification.
            model2: Secondary model specification when `compare_same_model` is False.
            compare_same_model: Whether to compare `model1` against itself.
            target_pd_mode: Target efficiency mode ('athena' or 'custom').
            custom_target_pd: Target detection efficiency used when `target_pd_mode == 'custom'`.
            working_point: Athena working point ('loose', 'medium', 'tight').
            data_path: Optional override path to consolidated `.npz` directory.
            manifest_path: Optional path to `split_manifest.json`.
            language: Plot localization language ('en' or 'pt').
            bins_mu: Bin count for pileup threshold calibration.
            n_bootstraps_pileup: Bootstrap count for pileup threshold calibration.
            plot_format: Output image format ('png' or 'pdf').
            log_y: Whether to use logarithmic y-scale on distribution histograms.
        """
        if target_pd_mode not in ("athena", "custom"):
            raise ValueError(f"Invalid target_pd_mode '{target_pd_mode}'. Expected 'athena' or 'custom'.")
        if not compare_same_model and model2 is None:
            raise ValueError("`model2` must be provided when `compare_same_model` is False.")

        self.iet = iet
        self.ieta = ieta
        self.region_key = f"et_{iet}_eta_{ieta}"
        self.model1 = model1
        self.model2 = model1 if compare_same_model else model2
        self.compare_same_model = compare_same_model
        self.target_pd_mode = target_pd_mode
        self.custom_target_pd = float(custom_target_pd)
        self.working_point = working_point.lower()
        self.data_path = data_path
        default_results = model1.results_path or (self.model2.results_path if self.model2 is not None else None)
        if manifest_path is None and default_results is None:
            raise ValueError("`manifest_path` must be provided when neither model specifies `results_path`.")
        self.manifest_path = manifest_path or (default_results.parent / "split_manifest.json")
        self.language = language
        self.bins_mu = bins_mu
        self.n_bootstraps_pileup = n_bootstraps_pileup
        self.plot_format = plot_format
        self.log_y = log_y
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _format_method_tag(self, method: str) -> str:
        if method == "athena_cut":
            return f"athena_cut_{self.working_point}"
        return method

    def build_comparison_tag(self, mode: str) -> str:
        """Builds a descriptive comparison tag reflecting any per-model method overrides."""
        if mode not in self.VALID_MODES:
            raise ValueError(f"Invalid mode '{mode}'. Expected one of {self.VALID_MODES}.")
        default_m1, default_m2 = self.MODE_TO_METHODS[mode]
        m1 = self._format_method_tag(self.model1.method or default_m1)
        m2 = self._format_method_tag(self.model2.method or default_m2)
        return f"{m1}_vs_{m2}"

    @staticmethod
    def discover_regions(manifest_path: Path) -> List[Tuple[int, int]]:
        """Discovers sorted (iet, ieta) region pairs from `split_manifest.json`."""
        data = SplitManifest(str(manifest_path)).load()
        regions: List[Tuple[int, int]] = []
        for key in data.keys():
            match = re.match(r"^et_(\d+)_eta_(\d+)$", str(key))
            if match:
                regions.append((int(match.group(1)), int(match.group(2))))
        return sorted(regions)

    @staticmethod
    def _sanitize_folder_token(name: str) -> str:
        cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", name.strip()).strip("_")
        return cleaned or "model"

    def _resolve_model_folder_name(self, spec: Optional[ComparisonModelConfig]) -> str:
        if spec is None:
            return "model"
        if spec.alias:
            return self._sanitize_folder_token(spec.alias)
        if spec.method == "athena_cut":
            return f"Athena_Cut_{self.working_point.capitalize()}"
        if spec.config_path is not None:
            return self._sanitize_folder_token(spec.config_path.stem)
        return "model"

    def _format_target_pd_tag(self) -> str:
        if self.target_pd_mode == "athena":
            return f"athena_{self.working_point}"
        pd_pct = f"{self.custom_target_pd * 100:.2f}".replace(".", "p")
        return f"custom_pd{pd_pct}"

    def build_output_dir(self, base_dir: Path, mode: str) -> Path:
        """Builds a descriptive output directory path for the active comparison configuration."""
        scope = "same" if self.compare_same_model else "cross"
        m1_folder = self._resolve_model_folder_name(self.model1)
        m2_folder = self._resolve_model_folder_name(self.model2)
        models_dir = f"{m1_folder}_vs_{m2_folder}"
        pd_tag = self._format_target_pd_tag()
        return base_dir / models_dir / f"Comparison_{self.build_comparison_tag(mode)}_{pd_tag}_{scope}_{self.region_key}"

    def _load_model_and_fold(
        self, spec: ComparisonModelConfig
    ) -> Tuple[NeuralRingerTrainer, torch.nn.Module, int]:
        with open(spec.config_path, "r") as f:
            cfg_dict = yaml.safe_load(f)
        if self.data_path is not None and self.data_path.exists():
            cfg_dict["dataset"]["parameters"]["drive_path"] = str(self.data_path)
        cfg = NeuralRingerTrainerConfiguration.model_validate(cfg_dict)
        trainer = NeuralRingerTrainer(cfg)
        agg = ResultAggregator(spec.results_path)
        best_dict, _, _, _ = agg.get_best_model_for_region(self.iet, self.ieta, cache=False)
        model = trainer.factory.create_model()
        model.load_state_dict(best_dict["best_weights"])
        model.to(self.device).eval()
        return trainer, model, int(best_dict.get("fold", 0))

    def _evaluate_athena_cut(
        self, raw_features: np.ndarray, feature_names: np.ndarray
    ) -> np.ndarray:
        grid = UserKinematicGrid()
        cut_eval = FastPhotonCutEvaluator(grid=grid)
        return cut_eval.evaluate_events_for_threshold(
            data=raw_features,
            feature_names=[str(f) for f in feature_names],
            threshold=grid.get_athena_threshold_for_et_bin(self.iet),
            working_point=self.working_point,
        )

    def _resolve_target_pd(
        self,
        raw_features: np.ndarray,
        raw_labels: np.ndarray,
        feature_names: np.ndarray,
        pass_athena: Optional[np.ndarray] = None,
    ) -> Tuple[float, str]:
        if self.target_pd_mode == "athena":
            if pass_athena is None:
                pass_athena = self._evaluate_athena_cut(raw_features, feature_names)
            target_pd = float(np.clip(np.mean(pass_athena[raw_labels == 1]), 1e-6, 1.0 - 1e-6))
            return target_pd, f"Athena {self.working_point.capitalize()} PD={target_pd * 100:.2f}%"
        return self.custom_target_pd, f"PD={self.custom_target_pd * 100:.2f}%"

    def _compute_best_sp_predictions(
        self,
        model: torch.nn.Module,
        raw_features: np.ndarray,
        raw_labels: np.ndarray,
        val_indices: np.ndarray,
        ring_cols: np.ndarray,
        eval_res: HoldoutEvaluationResult,
    ) -> Tuple[np.ndarray, float]:
        val_logits = HoldoutEvaluator.predict_logits(
            model, raw_features[np.ix_(val_indices, ring_cols)], self.device
        )
        val_probs = 1.0 / (1.0 + np.exp(-val_logits))
        fpr_v, tpr_v, thr_v = roc_curve(raw_labels[val_indices], val_probs)
        sp_v = np.array([compute_sp_metric(float(pd_i), float(fa_i)) for pd_i, fa_i in zip(tpr_v, fpr_v)])
        best_sp_thresh = float(thr_v[int(np.argmax(sp_v))])
        preds = (eval_res.predicted_probabilities >= best_sp_thresh).astype(int)
        return preds, best_sp_thresh

    def _compute_pileup_predictions(
        self,
        model: torch.nn.Module,
        raw_features: np.ndarray,
        raw_labels: np.ndarray,
        feature_names: np.ndarray,
        train_fold: Tuple[np.ndarray, np.ndarray],
        test_indices: np.ndarray,
        ring_cols: np.ndarray,
        eval_res: HoldoutEvaluationResult,
        target_pd: float,
    ) -> Tuple[np.ndarray, float, Any, np.ndarray, np.ndarray]:
        tr_idx, va_idx = train_fold
        calib_idx = np.concatenate([tr_idx, va_idx])
        logits_cal = HoldoutEvaluator.predict_logits(
            model, raw_features[np.ix_(calib_idx, ring_cols)], self.device
        )
        mu_all = PileupAnalyzer.extract_feature(raw_features, [str(f) for f in feature_names], "avgmu")
        sig_mask = raw_labels[calib_idx] == 1
        mu_sig_cal = mu_all[calib_idx][sig_mask]
        logits_sig_cal = logits_cal[sig_mask]
        cal = PileupThresholdCalibrator(
            bins_mu=self.bins_mu, n_bootstraps=self.n_bootstraps_pileup, random_seed=42
        ).calibrate_working_point(
            mu_values=mu_sig_cal,
            discriminant_values=logits_sig_cal,
            target_pd=target_pd,
            working_point_name="Target",
        )
        preds = PileupThresholdCalibrator.apply_threshold(
            eval_res.predicted_logits, mu_all[test_indices], cal.slope, cal.intercept
        )
        return preds, float(cal.intercept), cal, mu_sig_cal, logits_sig_cal

    def _resolve_strategy_predictions(
        self,
        method: str,
        spec: ComparisonModelConfig,
        model: Optional[torch.nn.Module],
        best_fold: int,
        ring_cols: Optional[np.ndarray],
        eval_res: Optional[HoldoutEvaluationResult],
        raw_features: np.ndarray,
        raw_labels: np.ndarray,
        feature_names: np.ndarray,
        train_folds: List[Tuple[np.ndarray, np.ndarray]],
        test_indices: np.ndarray,
        target_pd: float,
        pd_tag: str,
        preds_athena: np.ndarray,
    ) -> Tuple[np.ndarray, float, str, Optional[np.ndarray], bool, Optional[Dict[str, Any]]]:
        if method == "athena_cut":
            name = spec.alias or f"Athena Cut ({self.working_point.capitalize()})"
            return preds_athena, 0.0, name, None, False, None
        assert model is not None and ring_cols is not None and eval_res is not None
        if method == "best_sp":
            preds, thresh = self._compute_best_sp_predictions(
                model, raw_features, raw_labels, train_folds[best_fold][1], ring_cols, eval_res
            )
            return preds, thresh, f"{spec.alias} [Best SP]", eval_res.predicted_probabilities, True, None
        if method == "hard_cut":
            return (
                eval_res.predictions_calibrated_cut,
                float(eval_res.threshold),
                f"{spec.alias} [Hard Cut, {pd_tag}]",
                eval_res.predicted_probabilities,
                True,
                None,
            )
        preds, thresh, cal, mu_sig_cal, logits_sig_cal = self._compute_pileup_predictions(
            model, raw_features, raw_labels, feature_names, train_folds[best_fold],
            test_indices, ring_cols, eval_res, target_pd,
        )
        s_name = f"{spec.alias} [T(mu), {pd_tag}]"
        cal_info = {
            "strategy_name": s_name,
            "calibration": cal,
            "mu_signal_calib": mu_sig_cal,
            "logits_signal_calib": logits_sig_cal,
        }
        return preds, thresh, s_name, eval_res.predicted_probabilities, True, cal_info

    def prepare(self, mode: str, features_to_plot: List[str]) -> ComparisonPayload:
        """Loads models/data and computes predictions for the requested comparison mode.

        Args:
            mode: One of 'best_sp_vs_hard_cut', 'best_sp_vs_pileup', 'hard_cut_vs_pileup',
                'best_sp_vs_best_sp', or 'pileup_vs_pileup'.
            features_to_plot: Candidate feature names to extract for distribution plotting.

        Returns:
            ComparisonPayload ready for quadrant and variable distribution analysis.
        """
        if mode not in self.VALID_MODES:
            raise ValueError(f"Invalid mode '{mode}'. Expected one of {self.VALID_MODES}.")

        default_m1, default_m2 = self.MODE_TO_METHODS[mode]
        method_1 = self.model1.method or default_m1
        method_2 = self.model2.method or default_m2
        if method_1 == "athena_cut" and method_2 == "athena_cut":
            raise ValueError("At least one strategy must be a trained model.")

        trainer_1 = model_1 = best_fold_1 = None
        if method_1 != "athena_cut":
            trainer_1, model_1, best_fold_1 = self._load_model_and_fold(self.model1)

        trainer_2 = model_2 = best_fold_2 = None
        if not self.compare_same_model and method_2 != "athena_cut":
            trainer_2, model_2, best_fold_2 = self._load_model_and_fold(self.model2)

        ref_trainer = trainer_1 if trainer_1 is not None else trainer_2
        assert ref_trainer is not None
        full_dataset = ref_trainer.full_dataset
        feature_names = full_dataset.feature_names

        region_idx = next(
            (i for i, fp in enumerate(full_dataset.file_paths) if get_et_eta(fp) == (self.iet, self.ieta)),
            None,
        )
        if region_idx is None:
            raise FileNotFoundError(f"No dataset file matching et={self.iet}, eta={self.ieta}.")

        raw_features, raw_labels, _ = full_dataset[region_idx]
        manifest = SplitManifest(str(self.manifest_path))
        manifest.load()
        test_indices, train_folds = manifest.get_region_splits(self.region_key)
        test_indices_arr = np.asarray(test_indices, dtype=int)
        test_features = raw_features[test_indices_arr]
        test_labels = raw_labels[test_indices_arr]

        pass_athena = self._evaluate_athena_cut(raw_features, feature_names)
        preds_athena = pass_athena[test_indices_arr].astype(int)
        target_pd, pd_tag = self._resolve_target_pd(
            raw_features, raw_labels, feature_names, pass_athena=pass_athena
        )
        evaluator = HoldoutEvaluator(default_target_pd=target_pd)

        ring_cols_1 = eval_1 = None
        if trainer_1 is not None and model_1 is not None:
            trainer_1.full_dataset._set_ring_column_indices(feature_names)
            ring_cols_1 = trainer_1.full_dataset.ring_column_indices
            eval_1 = evaluator.evaluate(
                model_1, raw_features, raw_labels, test_indices_arr, ring_cols_1, self.device, target_pd=target_pd
            )

        if self.compare_same_model:
            model_2, best_fold_2, ring_cols_2, eval_2 = model_1, best_fold_1, ring_cols_1, eval_1
        elif trainer_2 is not None and model_2 is not None:
            trainer_2.full_dataset._set_ring_column_indices(feature_names)
            ring_cols_2 = trainer_2.full_dataset.ring_column_indices
            eval_2 = evaluator.evaluate(
                model_2, raw_features, raw_labels, test_indices_arr, ring_cols_2, self.device, target_pd=target_pd
            )
        else:
            ring_cols_2 = eval_2 = None

        preds_s1, thresh_s1, s1_name, probs_s1, is_s1_ml, cal_1 = self._resolve_strategy_predictions(
            method_1, self.model1, model_1, best_fold_1 or 0, ring_cols_1, eval_1,
            raw_features, raw_labels, feature_names, train_folds, test_indices_arr,
            target_pd, pd_tag, preds_athena,
        )
        preds_s2, thresh_s2, s2_name, probs_s2, is_s2_ml, cal_2 = self._resolve_strategy_predictions(
            method_2, self.model2, model_2, best_fold_2 or 0, ring_cols_2, eval_2,
            raw_features, raw_labels, feature_names, train_folds, test_indices_arr,
            target_pd, pd_tag, preds_athena,
        )

        pileup_cals: List[Dict[str, Any]] = []
        if cal_1 is not None:
            cal_1["tag"] = "s1"
            pileup_cals.append(cal_1)
        if cal_2 is not None:
            cal_2["tag"] = "s2"
            pileup_cals.append(cal_2)

        try:
            mu_test: Optional[np.ndarray] = PileupAnalyzer.extract_feature(
                test_features, [str(f) for f in feature_names], "avgmu"
            )
        except KeyError:
            mu_test = None

        valid_features: List[str] = []
        for feat in features_to_plot:
            try:
                extract_feature_values(test_features[:5], feature_names, feat)
                valid_features.append(feat)
            except KeyError:
                continue

        showershapes = {
            feat: extract_feature_values(test_features, feature_names, feat)
            for feat in valid_features
            if feat not in ("trig_L2_calo_et", "trig_L2_calo_eta", "trig_L2_calo_phi")
        }
        et_vals = extract_feature_values(test_features, feature_names, "trig_L2_calo_et")
        eta_vals = extract_feature_values(test_features, feature_names, "trig_L2_calo_eta")

        return ComparisonPayload(
            mode=self.build_comparison_tag(mode),
            region_key=self.region_key,
            iet=self.iet,
            ieta=self.ieta,
            strategy1_name=s1_name,
            strategy2_name=s2_name,
            preds_s1=preds_s1,
            preds_s2=preds_s2,
            thresh_s1=thresh_s1,
            thresh_s2=thresh_s2,
            probs_s1=probs_s1,
            probs_s2=probs_s2,
            test_features=test_features,
            test_labels=test_labels,
            feature_names=feature_names,
            valid_features=valid_features,
            showershapes=showershapes,
            et_vals=et_vals,
            eta_vals=eta_vals,
            preds_athena=preds_athena,
            athena_name=f"Athena Cut ({self.working_point.capitalize()})",
            is_s1_ml=is_s1_ml,
            is_s2_ml=is_s2_ml,
            mu_test=mu_test,
            pileup_calibrations=pileup_cals,
        )

    def run_quadrant_analysis(
        self, payload: ComparisonPayload, output_dir: Path, tables_dir: Path
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, List[str]]]:
        """Computes quadrant metrics/tables and exports separated signal/background plots."""
        s1_m = StrategyMetrics.from_predictions(payload.preds_s1, payload.test_labels)
        s2_m = StrategyMetrics.from_predictions(payload.preds_s2, payload.test_labels)
        qm = QuadrantAnalyzer.compute_quadrant_metrics(payload.preds_s1, payload.preds_s2, payload.test_labels)

        quad_res = RegionalQuadrantResult(
            iet=payload.iet,
            ieta=payload.ieta,
            strategy1_name=payload.strategy1_name,
            strategy2_name=payload.strategy2_name,
            metrics_strategy1=s1_m,
            metrics_strategy2=s2_m,
            overall_metrics=qm,
            signal_metrics=QuadrantAnalyzer.compute_quadrant_metrics(
                payload.preds_s1[payload.test_labels == 1],
                payload.preds_s2[payload.test_labels == 1],
                payload.test_labels[payload.test_labels == 1],
            ),
            background_metrics=QuadrantAnalyzer.compute_quadrant_metrics(
                payload.preds_s1[payload.test_labels == 0],
                payload.preds_s2[payload.test_labels == 0],
                payload.test_labels[payload.test_labels == 0],
            ),
            preds_strategy1=payload.preds_s1,
            preds_strategy2=payload.preds_s2,
            labels=payload.test_labels,
            threshold_strategy1=payload.thresh_s1,
            threshold_strategy2=payload.thresh_s2,
            showershapes=payload.showershapes,
            probs_model1=payload.probs_s1,
            probs_model2=payload.probs_s2,
            et=payload.et_vals,
            eta=payload.eta_vals,
        )

        strategy_df = pd.DataFrame([
            {
                "Strategy": payload.strategy1_name,
                "PD": s1_m.pd, "PD_unc": s1_m.pd_uncertainty,
                "PF": s1_m.pf, "PF_unc": s1_m.pf_uncertainty,
                "SP": s1_m.sp, "SP_unc": s1_m.sp_uncertainty,
            },
            {
                "Strategy": payload.strategy2_name,
                "PD": s2_m.pd, "PD_unc": s2_m.pd_uncertainty,
                "PF": s2_m.pf, "PF_unc": s2_m.pf_uncertainty,
                "SP": s2_m.sp, "SP_unc": s2_m.sp_uncertainty,
            },
        ])
        counts_df = pd.DataFrame({
            "Quadrant": [
                "Both Correct",
                f"Only {payload.strategy1_name} Correct",
                f"Only {payload.strategy2_name} Correct",
                "Both Wrong",
            ],
            "Count": [qm.both_correct, qm.model1_only_correct, qm.model2_only_correct, qm.both_wrong],
            "Ratio": [
                f"{qm.both_correct_ratio:.2%}",
                f"{qm.model1_only_correct_ratio:.2%}",
                f"{qm.model2_only_correct_ratio:.2%}",
                f"{qm.both_wrong_ratio:.2%}",
            ],
        })
        save_table(strategy_df, tables_dir, payload.region_key, f"{payload.mode}_quadrant_strategy_metrics")
        save_table(counts_df, tables_dir, payload.region_key, f"{payload.mode}_quadrant_counts")

        plotter = QuadrantPlotter(language=self.language)
        quad_prefix = f"{payload.region_key}_{payload.mode}_quadrant_plot"
        saved_plots = plotter.plot(
            [quad_res],
            output_dir / "Quadrant",
            file_format=self.plot_format,
            log_y=self.log_y,
            file_prefix=quad_prefix,
        )
        return strategy_df, counts_df, saved_plots

    def run_variable_analysis(
        self,
        payload: ComparisonPayload,
        output_dir: Path,
        tables_dir: Path,
        is_signal: bool,
        include_athena_reference: bool = True,
    ) -> Tuple[pd.DataFrame, Any]:
        """Computes JS divergence table and exports feature distribution plots for one class."""
        cls_val = 1 if is_signal else 0
        strategies = [
            {
                "name": payload.strategy1_name,
                "preds": payload.preds_s1,
                "is_ml": payload.is_s1_ml,
                "color": "#005A9C",
                "marker": "s",
            },
            {
                "name": payload.strategy2_name,
                "preds": payload.preds_s2,
                "is_ml": payload.is_s2_ml,
                "color": "#E65100",
                "marker": "^",
            },
        ]
        if (
            include_athena_reference
            and payload.preds_athena is not None
            and payload.is_s1_ml
            and payload.is_s2_ml
        ):
            strategies.append({
                "name": payload.athena_name or f"Athena Cut ({self.working_point.capitalize()})",
                "preds": payload.preds_athena,
                "is_ml": False,
                "color": "#2E7D32",
                "marker": "o",
            })
        plot_data: Dict[str, Any] = {}
        for feat_name in payload.valid_features:
            vals = extract_feature_values(payload.test_features, payload.feature_names, feat_name)
            v_true = vals[payload.test_labels == cls_val]
            v_true = v_true[np.isfinite(v_true)]
            bins = np.linspace(np.percentile(v_true, 0.5), np.percentile(v_true, 99.5), 35)
            c_true, _ = np.histogram(v_true, bins=bins)
            models_list = []
            for m in strategies:
                v_m = vals[(payload.test_labels == cls_val) & (m["preds"] == 1)]
                c_m, _ = np.histogram(v_m[np.isfinite(v_m)], bins=bins)
                models_list.append({
                    "name": m["name"], "counts": c_m, "errors": np.sqrt(c_m),
                    "is_ml": m["is_ml"], "color": m["color"], "marker": m["marker"], "label": m["name"],
                })
            plot_data[feat_name] = {
                "display_title": feat_name,
                "bin_centers": 0.5 * (bins[:-1] + bins[1:]),
                "true_counts": c_true,
                "true_n": len(v_true),
                "models": models_list,
            }

        comparator = DistributionComparator(
            n_bootstraps=100, random_seed=42, compute_ks=False, compute_wasserstein=False, compute_kl=False, compute_js=True
        )
        js_df = comparator.compare_strategies(
            payload.test_features,
            payload.feature_names,
            payload.test_labels,
            payload.valid_features,
            {m["name"]: m["preds"] for m in strategies},
            reference_strategy=None,
            is_signal=is_signal,
        )
        suffix = "signal" if is_signal else "background"
        save_table(js_df, tables_dir, payload.region_key, f"{payload.mode}_variable_js_divergence_{suffix}")

        has_athena_ref = (
            include_athena_reference
            and payload.preds_athena is not None
            and payload.is_s1_ml
            and payload.is_s2_ml
        )
        ref_tag = f"_with_athena_cut_{self.working_point}_ref" if has_athena_ref else ""
        file_prefix = f"{payload.region_key}_{payload.mode}{ref_tag}_variable_plot"

        plotter = FeatureDistributionPlotter(language=self.language)
        fig = plotter.plot_and_export_all(
            plot_data,
            payload.valid_features,
            output_dir / f"Variables_{suffix.capitalize()}",
            region_key=payload.region_key,
            is_signal=is_signal,
            log_y=self.log_y,
            plot_format=self.plot_format,
            file_prefix=file_prefix,
            save_overview_grid=True,
        )
        return js_df, fig

    def run_pileup_analysis(
        self,
        payload: ComparisonPayload,
        output_dir: Path,
        tables_dir: Path,
        include_athena_reference: bool = True,
    ) -> Tuple[pd.DataFrame, Dict[str, List[Any]]]:
        """Exports 2D pileup threshold calibration plots and efficiency-vs-mu plots."""
        pileup_dir = output_dir / "Pileup"
        pileup_dir.mkdir(parents=True, exist_ok=True)
        tables_dir.mkdir(parents=True, exist_ok=True)

        plotter = PileupPlotter(language=self.language)
        calibrator = PileupThresholdCalibrator(
            bins_mu=self.bins_mu, n_bootstraps=self.n_bootstraps_pileup, random_seed=42
        )

        fit_rows: List[Dict[str, Any]] = []
        calibration_figs: List[Any] = []
        for cal_info in payload.pileup_calibrations or []:
            cal = cal_info["calibration"]
            tag = cal_info.get("tag", "s")
            dist_2d = calibrator.compute_2d_distribution(
                cal_info["mu_signal_calib"], cal_info["logits_signal_calib"], bins=35
            )
            fig_calib = plotter.plot_2d_threshold_calibration(
                distribution=dist_2d,
                calibrations=[cal],
                title="",
                region_label=f"iet{payload.iet}.ieta{payload.ieta}",
                output_path=(
                    pileup_dir
                    / f"{payload.region_key}_{payload.mode}_pileup_2d_threshold_calibration_{tag}_plot.{self.plot_format}"
                ),
                file_format=self.plot_format,
            )
            calibration_figs.append(fig_calib)
            sign = "+" if cal.intercept >= 0 else "-"
            fit_rows.append({
                "Strategy": cal_info["strategy_name"],
                "Target PD (%)": f"{cal.target_pd * 100:.2f}%",
                "Slope (a)": f"{cal.slope:.4f}",
                "Intercept (b)": f"{cal.intercept:.4f}",
                "R^2": f"{cal.r_squared:.4f}",
                "Fit Formula": f"T(mu) = {cal.slope:.4f}*mu {sign} {abs(cal.intercept):.4f}",
            })

        fit_df = pd.DataFrame(fit_rows)
        if not fit_df.empty:
            save_table(fit_df, tables_dir, payload.region_key, f"{payload.mode}_pileup_fit_parameters")

        efficiency_figs: List[Any] = []
        if payload.mu_test is not None:
            strategies = [
                {
                    "name": payload.strategy1_name,
                    "preds": payload.preds_s1,
                    "is_ml": payload.is_s1_ml,
                    "color": "#005A9C",
                    "marker": "s",
                    "is_reference": not payload.is_s1_ml,
                },
                {
                    "name": payload.strategy2_name,
                    "preds": payload.preds_s2,
                    "is_ml": payload.is_s2_ml,
                    "color": "#E65100",
                    "marker": "^",
                    "is_reference": not payload.is_s2_ml,
                },
            ]
            if (
                include_athena_reference
                and payload.preds_athena is not None
                and payload.is_s1_ml
                and payload.is_s2_ml
            ):
                strategies.append({
                    "name": payload.athena_name or f"Athena Cut ({self.working_point.capitalize()})",
                    "preds": payload.preds_athena,
                    "is_ml": False,
                    "color": "#2E7D32",
                    "marker": "o",
                    "is_reference": True,
                })

            for is_sig, cls_tag in [(True, "signal"), (False, "background")]:
                models_data: List[Dict[str, Any]] = []
                table_cols: Dict[str, Any] = {}
                for idx_s, s in enumerate(strategies):
                    eff = PileupAnalyzer.compute_efficiency_vs_mu(
                        payload.mu_test, payload.test_labels, s["preds"], bins=self.bins_mu, is_signal=is_sig
                    )
                    if idx_s == 0:
                        table_cols["mu_bin_center"] = eff["bin_centers"]
                    table_cols[f"{s['name']}_eff"] = eff["efficiencies"]
                    table_cols[f"{s['name']}_unc"] = eff["uncertainties"]
                    models_data.append({
                        "label": s["name"],
                        "bin_centers": eff["bin_centers"],
                        "efficiencies": eff["efficiencies"],
                        "uncertainties": eff["uncertainties"],
                        "color": s["color"],
                        "marker": s["marker"],
                        "is_ml": s["is_ml"],
                        "is_reference": s["is_reference"],
                    })

                save_table(
                    pd.DataFrame(table_cols),
                    tables_dir,
                    payload.region_key,
                    f"{payload.mode}_pileup_efficiency_vs_mu_{cls_tag}",
                )
                fig_eff = plotter.plot_efficiency_vs_mu(
                    models_data=models_data,
                    title="",
                    region_label=f"iet{payload.iet}.ieta{payload.ieta}",
                    is_signal=is_sig,
                    output_path=(
                        pileup_dir
                        / f"{payload.region_key}_{payload.mode}_pileup_efficiency_vs_mu_{cls_tag}_plot.{self.plot_format}"
                    ),
                    file_format=self.plot_format,
                )
                efficiency_figs.append(fig_eff)

        return fit_df, {"calibration_figs": calibration_figs, "efficiency_figs": efficiency_figs}
