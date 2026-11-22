import gc
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import pandas as pd
import torch
import yaml

from src.core.Datasets.SplitManifest import SplitManifest
from src.core.Trainers.NeuralRingerTrainer import NeuralRingerTrainer
from src.core.Validation.FastPhotonCutEvaluator import (
    FastPhotonCutEvaluator,
    UserKinematicGrid,
)
from src.core.Validation.HoldoutEvaluator import HoldoutEvaluator
from src.core.Validation.LatexTableFormatter import LatexTableFormatter
from src.core.Validation.PileupAnalyzer import PileupAnalyzer
from src.core.Validation.PileupThresholdCalibrator import PileupThresholdCalibrator
from src.core.Validation.QuadrantAnalyzer import StrategyMetrics
from src.core.Validation.ResultAggregator import ResultAggregator
from src.Parser.NeuralRingerTrainerConfiguration import NeuralRingerTrainerConfiguration
from src.utils import get_et_eta

log = logging.getLogger("RegionalOperatingPointsConsolidator")


class RegionalOperatingPointsConsolidator:
    """Consolidates operating point performance metrics across all kinematic regions."""

    ET_LABELS_LATEX: Dict[int, str] = {
        0: "$15 \\le E_T < 20\\,\\text{GeV}$",
        1: "$20 \\le E_T < 30\\,\\text{GeV}$",
        2: "$30 \\le E_T < 40\\,\\text{GeV}$",
        3: "$40 \\le E_T < 50\\,\\text{GeV}$",
        4: "$E_T \\ge 50\\,\\text{GeV}$",
    }

    ETA_LABELS_LATEX: Dict[int, str] = {
        0: "$0.00 \\le |\\eta| < 0.80$",
        1: "$0.80 \\le |\\eta| < 1.37$",
        2: "$1.37 \\le |\\eta| < 1.54$",
        3: "$1.54 \\le |\\eta| < 2.37$",
        4: "$2.37 \\le |\\eta| < 2.50$",
    }

    DEFAULT_METHOD_KEYS: Tuple[str, str, str] = (
        "calibrated_hard_cut",
        "pileup_corrected_cut",
        "athena_cut",
    )
    TRAINING_METHOD_KEY: str = "training_best_sp"

    def __init__(
        self,
        config_path: Union[str, Path],
        results_path: Union[str, Path],
        manifest_path: Optional[Union[str, Path]] = None,
        target_pd: float = 0.9424,
        working_point: str = "loose",
        bins_mu: int = 10,
        n_bootstraps_pileup: int = 100,
        efficiencies_csv: Optional[Union[str, Path]] = None,
        device: Optional[Union[str, torch.device]] = None,
        method_aliases: Optional[Dict[str, str]] = None,
    ) -> None:
        """Initializes the consolidator with configuration, data paths, and parameters.

        Args:
            config_path: Path to NeuralRinger YAML configuration.
            results_path: Path to model training results directory.
            manifest_path: Optional path to split manifest JSON file.
            target_pd: Target signal detection efficiency.
            working_point: Athena fast photon cut working point ('loose', 'medium', 'tight').
            bins_mu: Bin count for pileup threshold calibration.
            n_bootstraps_pileup: Bootstrap sample count for pileup threshold fitting.
            efficiencies_csv: Optional path to reference efficiencies CSV.
            device: Computing device ('cpu', 'cuda', or torch.device).
            method_aliases: Optional display name mapping for methods.
        """
        self.config_path: Path = Path(config_path)
        self.results_path: Path = Path(results_path)
        self.target_pd: float = float(target_pd)
        self.working_point: str = working_point.lower()
        self.bins_mu: int = bins_mu
        self.n_bootstraps_pileup: int = n_bootstraps_pileup

        if isinstance(config_path, NeuralRingerTrainerConfiguration):
            self.config = config_path
            self.config_path = Path("in_memory_config")
        else:
            self.config_path = Path(config_path)
            self.config = self._load_config(self.config_path)

        self.trainer: NeuralRingerTrainer = NeuralRingerTrainer(self.config)

        if manifest_path is not None:
            resolved_manifest = Path(manifest_path)
        elif (self.results_path / "split_manifest.json").exists():
            resolved_manifest = self.results_path / "split_manifest.json"
        elif (self.results_path.parent / "split_manifest.json").exists():
            resolved_manifest = self.results_path.parent / "split_manifest.json"
        else:
            resolved_manifest = self.results_path / "split_manifest.json"

        self.manifest: SplitManifest = SplitManifest(str(resolved_manifest))
        self.aggregator: ResultAggregator = ResultAggregator(self.results_path)

        self.device: torch.device = (
            torch.device(device)
            if device is not None
            else self.trainer.device
        )
        self.evaluator: HoldoutEvaluator = HoldoutEvaluator(default_target_pd=self.target_pd)
        self.grid: UserKinematicGrid = UserKinematicGrid()
        self.cut_evaluator: FastPhotonCutEvaluator = FastPhotonCutEvaluator(grid=self.grid)

        self.reference_eff_df: Optional[pd.DataFrame] = None
        if efficiencies_csv is not None and Path(efficiencies_csv).exists():
            self.reference_eff_df = pd.read_csv(Path(efficiencies_csv))

        self.method_aliases: Dict[str, str] = {
            "calibrated_hard_cut": f"Calibrated Hard Cut (Target PD: {self.target_pd*100:.2f}%)",
            "pileup_corrected_cut": f"Pileup-Corrected Cut T(mu) (Target PD)",
            "athena_cut": f"Athena {self.working_point.capitalize()} Cut",
            "training_best_sp": "Training Best SP",
        }
        if method_aliases:
            self.method_aliases.update(method_aliases)

        self._records: List[Dict[str, Any]] = []
        self._training_records: List[Dict[str, Any]] = []
        self._athena_matched_records: List[Dict[str, Any]] = []

    @property
    def records(self) -> List[Dict[str, Any]]:
        """Returns the list of extracted regional evaluation records."""
        return self._records

    @staticmethod
    def _load_config(config_path: Union[str, Path]) -> NeuralRingerTrainerConfiguration:
        with open(config_path, "r") as f:
            yaml_data = yaml.safe_load(f)
        return NeuralRingerTrainerConfiguration.model_validate(yaml_data)

    def _get_target_pd_for_region(self, iet: int, ieta: int) -> float:
        if self.reference_eff_df is not None:
            col_name = f"eff_sig_{self.working_point}"
            sub = self.reference_eff_df[
                (self.reference_eff_df["et_bin"] == iet)
                & (self.reference_eff_df["eta_bin"] == ieta)
            ]
            if not sub.empty and col_name in sub.columns:
                return float(sub[col_name].iloc[0])
        return self.target_pd

    @staticmethod
    def _strategy_metrics_record(
        iet: int,
        ieta: int,
        method_key: str,
        method_name: str,
        target_pd: float,
        decision_threshold: str,
        metrics: StrategyMetrics,
    ) -> Dict[str, Any]:
        return {
            "iet": iet,
            "ieta": ieta,
            "method_key": method_key,
            "method": method_name,
            "target_pd": target_pd,
            "decision_threshold": decision_threshold,
            "pd": metrics.pd,
            "pd_uncertainty": metrics.pd_uncertainty,
            "sp": metrics.sp,
            "sp_uncertainty": metrics.sp_uncertainty,
            "pf": metrics.pf,
            "pf_uncertainty": metrics.pf_uncertainty,
            "eff": metrics.eff,
            "eff_uncertainty": metrics.eff_uncertainty,
            "bg_eff": metrics.bg_eff,
            "bg_eff_uncertainty": metrics.bg_eff_uncertainty,
            "tp": metrics.tp,
            "fp": metrics.fp,
            "tn": metrics.tn,
            "fn": metrics.fn,
            "total": metrics.total,
        }

    def scan_and_evaluate(self) -> List[Dict[str, Any]]:
        """Scans all available dataset regions and extracts performance for all three methods.

        Returns:
            List of record dictionaries containing regional metrics.
        """
        if not self.manifest.exists():
            raise FileNotFoundError(
                f"Split manifest not found at {self.manifest.manifest_file}."
            )
        self.manifest.load()

        model = self.trainer.factory.create_model().to(self.device)
        total_regions = len(self.trainer.full_dataset)
        self._records = []

        log.info(f"Scanning {total_regions} regions from dataset for operating points evaluation...")

        for index in range(total_regions):
            data_path = self.trainer.full_dataset.file_paths[index]
            iet, ieta = get_et_eta(data_path)

            if not self.config.is_region_allowed(iet, ieta):
                log.info(f"[iet{iet}.ieta{ieta}] Skipped: excluded by configuration filters.")
                continue

            if not self.aggregator.has_region(iet, ieta):
                log.info(f"[iet{iet}.ieta{ieta}] Skipped: no trained results in {self.results_path}.")
                continue

            region_key = f"et_{iet}_eta_{ieta}"
            if region_key not in self.manifest.data:
                log.warning(f"[iet{iet}.ieta{ieta}] Skipped: region key not in split manifest.")
                continue

            test_indices, train_folds = self.manifest.get_region_splits(region_key)
            test_indices_arr = np.asarray(test_indices, dtype=int)
            if len(test_indices_arr) == 0:
                log.warning(f"[iet{iet}.ieta{ieta}] Skipped: empty holdout test set.")
                continue

            try:
                best_model, best_rep, mean_sp, std_sp = self.aggregator.get_best_model_for_region(
                    iet, ieta, cache=False
                )
                model.load_state_dict(best_model["best_weights"])
                best_fold = int(best_model.get("fold", 0))
                del best_model
                model.eval()
            except Exception as exc:
                log.warning(f"[iet{iet}.ieta{ieta}] Failed to load best model: {exc}")
                continue

            raw_features, raw_labels, _ = self.trainer.full_dataset[index]
            ring_indices = self.trainer.full_dataset.ring_column_indices
            if ring_indices is None:
                ring_indices = getattr(self.trainer.full_dataset, "indexes", None)

            feature_names = getattr(self.trainer.full_dataset, "feature_names", None)
            feature_names_list = [str(f) for f in feature_names] if feature_names is not None else []

            target_pd = self._get_target_pd_for_region(iet, ieta)

            eval_res = self.evaluator.evaluate(
                model=model,
                data=raw_features,
                target=raw_labels,
                test_indices=test_indices_arr,
                ring_column_indices=ring_indices,
                device=self.device,
                target_pd=target_pd,
            )

            metrics_hard = StrategyMetrics.from_predictions(
                eval_res.predictions_calibrated_cut, eval_res.holdout_labels
            )

            train_idx, val_idx = train_folds[best_fold]
            calib_idx = np.concatenate([train_idx, val_idx])

            calib_ring_features = np.ascontiguousarray(
                raw_features[np.ix_(calib_idx, ring_indices)], dtype=np.float32
            )
            calib_labels = raw_labels[calib_idx]
            mu_all = PileupAnalyzer.extract_feature(raw_features, feature_names_list, "avgmu")
            mu_calib = mu_all[calib_idx]
            mu_test = mu_all[test_indices_arr]
            del raw_features, raw_labels, mu_all
            gc.collect()

            logits_calib = HoldoutEvaluator.predict_logits(
                model, calib_ring_features, self.device, batch_size=self.evaluator.batch_size
            )

            sig_calib_mask = calib_labels == 1
            calibrator = PileupThresholdCalibrator(
                bins_mu=self.bins_mu,
                n_bootstraps=self.n_bootstraps_pileup,
                random_seed=42,
            )
            cal_wp = calibrator.calibrate_working_point(
                mu_values=mu_calib[sig_calib_mask],
                discriminant_values=logits_calib[sig_calib_mask],
                target_pd=target_pd,
                working_point_name="Baseline",
            )
            preds_pileup = PileupThresholdCalibrator.apply_threshold(
                eval_res.predicted_logits, mu_test, cal_wp.slope, cal_wp.intercept
            )
            metrics_pileup = StrategyMetrics.from_predictions(preds_pileup, eval_res.holdout_labels)

            athena_th = self.grid.get_athena_threshold_for_et_bin(iet)
            pass_cut = self.cut_evaluator.evaluate_events_for_threshold(
                data=eval_res.holdout_features_full,
                feature_names=feature_names_list,
                threshold=athena_th,
                working_point=self.working_point,
            )
            preds_athena = pass_cut.astype(int)
            metrics_athena = StrategyMetrics.from_predictions(preds_athena, eval_res.holdout_labels)

            evaluated_methods = [
                (
                    "calibrated_hard_cut",
                    self.method_aliases["calibrated_hard_cut"],
                    metrics_hard,
                    f"{eval_res.threshold:.4f}",
                ),
                (
                    "pileup_corrected_cut",
                    self.method_aliases["pileup_corrected_cut"],
                    metrics_pileup,
                    f"T(mu)={cal_wp.slope:.4f}*mu+{cal_wp.intercept:.4f}",
                ),
                (
                    "athena_cut",
                    self.method_aliases["athena_cut"],
                    metrics_athena,
                    f"{athena_th:.1f} GeV",
                ),
            ]

            for key, name, met, th_repr in evaluated_methods:
                self._records.append(
                    self._strategy_metrics_record(
                        iet=iet,
                        ieta=ieta,
                        method_key=key,
                        method_name=name,
                        target_pd=target_pd,
                        decision_threshold=th_repr,
                        metrics=met,
                    )
                )

            log.info(
                f"[iet{iet}.ieta{ieta}] HardCut(SP={metrics_hard.sp:.4f}) | "
                f"PileupCut(SP={metrics_pileup.sp:.4f}) | Athena(SP={metrics_athena.sp:.4f})"
            )

            del (
                eval_res,
                calib_ring_features,
                calib_labels,
                logits_calib,
                mu_calib,
                mu_test,
                preds_pileup,
                pass_cut,
                preds_athena,
            )
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        log.info(f"Evaluation complete. Extracted {len(self._records)} method records.")
        return self._records

    def scan_training_best_sp(self) -> List[Dict[str, Any]]:
        """Scans regional training results and extracts metrics for the best-SP fold per region.

        Returns:
            List of record dictionaries containing best-SP training metrics per region.
        """
        total_regions = len(self.trainer.full_dataset)
        self._training_records = []

        for index in range(total_regions):
            data_path = self.trainer.full_dataset.file_paths[index]
            iet, ieta = get_et_eta(data_path)

            if not self.config.is_region_allowed(iet, ieta):
                continue
            if not self.aggregator.has_region(iet, ieta):
                continue

            try:
                metrics = self.aggregator.get_best_training_metrics_for_region(
                    iet, ieta, cache=False
                )
            except Exception as exc:
                log.warning(f"[iet{iet}.ieta{ieta}] Failed to load best training model: {exc}")
                continue

            self._training_records.append(
                {
                    "iet": iet,
                    "ieta": ieta,
                    "method_key": self.TRAINING_METHOD_KEY,
                    "method": self.method_aliases.get(self.TRAINING_METHOD_KEY, "Training Best SP"),
                    **metrics,
                }
            )

        log.info(f"Extracted {len(self._training_records)} regional training best-SP records.")
        return self._training_records

    def _athena_matched_method_aliases(self) -> Dict[str, str]:
        wp_label = self.working_point.capitalize()
        return {
            self.TRAINING_METHOD_KEY: self.method_aliases.get(
                self.TRAINING_METHOD_KEY, "Training Best SP"
            ),
            "calibrated_hard_cut": f"Calibrated Hard Cut (Athena {wp_label} PD)",
            "pileup_corrected_cut": f"Pileup-Corrected Cut T(mu) (Athena {wp_label} PD)",
            "athena_cut": self.method_aliases.get("athena_cut", f"Athena {wp_label} Cut"),
        }

    def scan_athena_matched_and_training(
        self,
        include_training_best_sp: bool = True,
    ) -> List[Dict[str, Any]]:
        """Evaluates neural cuts calibrated to full-dataset Athena PD (optionally with Training Best SP).

        Args:
            include_training_best_sp: Whether to include the Training Best SP record per region.

        Returns:
            List of record dictionaries containing the compared methods per region.
        """
        if not self.manifest.exists():
            raise FileNotFoundError(
                f"Split manifest not found at {self.manifest.manifest_file}."
            )
        self.manifest.load()

        model = self.trainer.factory.create_model().to(self.device)
        total_regions = len(self.trainer.full_dataset)
        self._athena_matched_records = []
        aliases = self._athena_matched_method_aliases()

        for index in range(total_regions):
            data_path = self.trainer.full_dataset.file_paths[index]
            iet, ieta = get_et_eta(data_path)

            if not self.config.is_region_allowed(iet, ieta):
                continue
            if not self.aggregator.has_region(iet, ieta):
                continue

            region_key = f"et_{iet}_eta_{ieta}"
            if region_key not in self.manifest.data:
                continue

            test_indices, train_folds = self.manifest.get_region_splits(region_key)
            test_indices_arr = np.asarray(test_indices, dtype=int)
            if len(test_indices_arr) == 0:
                continue

            try:
                train_metrics = (
                    self.aggregator.get_best_training_metrics_for_region(iet, ieta, cache=False)
                    if include_training_best_sp
                    else None
                )
                best_model, _, _, _ = self.aggregator.get_best_model_for_region(
                    iet, ieta, cache=False
                )
                model.load_state_dict(best_model["best_weights"])
                best_fold = int(best_model.get("fold", 0))
                del best_model
                model.eval()
            except Exception as exc:
                log.warning(f"[iet{iet}.ieta{ieta}] Failed to load best model: {exc}")
                continue

            raw_features, raw_labels, _ = self.trainer.full_dataset[index]
            ring_indices = self.trainer.full_dataset.ring_column_indices
            if ring_indices is None:
                ring_indices = getattr(self.trainer.full_dataset, "indexes", None)

            feature_names = getattr(self.trainer.full_dataset, "feature_names", None)
            feature_names_list = [str(f) for f in feature_names] if feature_names is not None else []

            athena_th = self.grid.get_athena_threshold_for_et_bin(iet)
            pass_cut_full = self.cut_evaluator.evaluate_events_for_threshold(
                data=raw_features,
                feature_names=feature_names_list,
                threshold=athena_th,
                working_point=self.working_point,
            )
            sig_mask_full = raw_labels == 1
            athena_pd_full = (
                float(np.mean(pass_cut_full[sig_mask_full]))
                if np.any(sig_mask_full)
                else self.target_pd
            )
            target_pd_athena = float(np.clip(athena_pd_full, 1e-6, 1.0 - 1e-6))

            preds_athena_test = pass_cut_full[test_indices_arr].astype(int)
            holdout_labels = raw_labels[test_indices_arr]
            metrics_athena = StrategyMetrics.from_predictions(preds_athena_test, holdout_labels)
            del pass_cut_full, sig_mask_full, preds_athena_test

            eval_res = self.evaluator.evaluate(
                model=model,
                data=raw_features,
                target=raw_labels,
                test_indices=test_indices_arr,
                ring_column_indices=ring_indices,
                device=self.device,
                target_pd=target_pd_athena,
            )
            metrics_hard = StrategyMetrics.from_predictions(
                eval_res.predictions_calibrated_cut, eval_res.holdout_labels
            )

            train_idx, val_idx = train_folds[best_fold]
            calib_idx = np.concatenate([train_idx, val_idx])
            calib_ring_features = np.ascontiguousarray(
                raw_features[np.ix_(calib_idx, ring_indices)], dtype=np.float32
            )
            calib_labels = raw_labels[calib_idx]
            mu_all = PileupAnalyzer.extract_feature(raw_features, feature_names_list, "avgmu")
            mu_calib = mu_all[calib_idx]
            mu_test = mu_all[test_indices_arr]
            del raw_features, raw_labels, mu_all
            gc.collect()

            logits_calib = HoldoutEvaluator.predict_logits(
                model, calib_ring_features, self.device, batch_size=self.evaluator.batch_size
            )
            sig_calib_mask = calib_labels == 1
            calibrator = PileupThresholdCalibrator(
                bins_mu=self.bins_mu,
                n_bootstraps=self.n_bootstraps_pileup,
                random_seed=42,
            )
            cal_wp = calibrator.calibrate_working_point(
                mu_values=mu_calib[sig_calib_mask],
                discriminant_values=logits_calib[sig_calib_mask],
                target_pd=target_pd_athena,
                working_point_name="Baseline",
            )
            preds_pileup = PileupThresholdCalibrator.apply_threshold(
                eval_res.predicted_logits, mu_test, cal_wp.slope, cal_wp.intercept
            )
            metrics_pileup = StrategyMetrics.from_predictions(preds_pileup, eval_res.holdout_labels)

            if include_training_best_sp and train_metrics is not None:
                self._athena_matched_records.append(
                    {
                        "iet": iet,
                        "ieta": ieta,
                        "method_key": self.TRAINING_METHOD_KEY,
                        "method": aliases[self.TRAINING_METHOD_KEY],
                        "target_pd": target_pd_athena,
                        "decision_threshold": "Best SP (Train)",
                        **train_metrics,
                    }
                )
            for key, met, th_repr in (
                ("calibrated_hard_cut", metrics_hard, f"{eval_res.threshold:.4f}"),
                (
                    "pileup_corrected_cut",
                    metrics_pileup,
                    f"T(mu)={cal_wp.slope:.4f}*mu+{cal_wp.intercept:.4f}",
                ),
                ("athena_cut", metrics_athena, f"{athena_th:.1f} GeV"),
            ):
                self._athena_matched_records.append(
                    self._strategy_metrics_record(
                        iet=iet,
                        ieta=ieta,
                        method_key=key,
                        method_name=aliases[key],
                        target_pd=target_pd_athena,
                        decision_threshold=th_repr,
                        metrics=met,
                    )
                )

            del (
                eval_res,
                calib_ring_features,
                calib_labels,
                logits_calib,
                mu_calib,
                mu_test,
                preds_pileup,
                holdout_labels,
            )
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        log.info(
            f"Athena-matched evaluation complete. Extracted {len(self._athena_matched_records)} records."
        )
        return self._athena_matched_records

    def build_tidy_dataframe(self) -> pd.DataFrame:
        """Returns the extracted metrics in long-form tidy DataFrame format."""
        if not self._records:
            self.scan_and_evaluate()
        return pd.DataFrame(self._records)

    def build_training_tidy_dataframe(self) -> pd.DataFrame:
        """Returns the extracted best-SP training metrics in long-form tidy DataFrame format."""
        if not self._training_records:
            self.scan_training_best_sp()
        return pd.DataFrame(self._training_records)

    def build_athena_matched_tidy_dataframe(
        self,
        include_training_best_sp: bool = True,
    ) -> pd.DataFrame:
        """Returns the Athena-matched comparison records (and optionally training best-SP) as a DataFrame."""
        if not self._athena_matched_records:
            self.scan_athena_matched_and_training(include_training_best_sp=include_training_best_sp)
        df = pd.DataFrame(self._athena_matched_records)
        if not include_training_best_sp and not df.empty and "method_key" in df.columns:
            df = df[df["method_key"] != self.TRAINING_METHOD_KEY].reset_index(drop=True)
        return df

    def _format_cell(
        self,
        value: float,
        uncertainty: float,
        precision: int = 2,
        decimal_comma: bool = False,
        math_delimiters: bool = True,
        bold: bool = False,
        allow_missing_err: bool = False,
    ) -> str:
        return LatexTableFormatter.format_value_with_uncertainty(
            val=value,
            err=uncertainty,
            precision=precision,
            bold=bold,
            decimal_comma=decimal_comma,
            math_delimiters=math_delimiters,
            nan_placeholder="-",
            allow_missing_err=allow_missing_err,
        )

    def _build_grid_from_dataframe(
        self,
        tidy_df: pd.DataFrame,
        method_keys: List[str],
        as_percentage: bool = True,
        precision: int = 2,
        decimal_comma: bool = False,
        math_delimiters: bool = True,
        bold_best_sp: bool = True,
        allow_missing_err: bool = False,
        method_label_overrides: Optional[Dict[str, str]] = None,
    ) -> pd.DataFrame:
        if tidy_df.empty:
            raise RuntimeError("No regional evaluation records available.")

        df = tidy_df.copy()
        scale = 100.0 if as_percentage else 1.0
        df["pd_disp"] = df["pd"] * scale
        df["pd_err_disp"] = df["pd_uncertainty"] * scale
        df["sp_disp"] = df["sp"] * scale
        df["sp_err_disp"] = df["sp_uncertainty"] * scale
        df["pf_disp"] = df["pf"] * scale
        df["pf_err_disp"] = df["pf_uncertainty"] * scale

        available_ets = sorted(df["iet"].unique().astype(int))
        available_etas = sorted(df["ieta"].unique().astype(int))

        metric_names = ["P_D [%]", "SP [%]", "P_F [%]"]
        col_tuples = []
        for iet in available_ets:
            et_label = self.ET_LABELS_LATEX.get(iet, f"$E_{{T{iet}}}$")
            for m in metric_names:
                col_tuples.append((et_label, m))
        multi_cols = pd.MultiIndex.from_tuples(col_tuples, names=["ET", "Metric"])

        index_tuples = []
        rows_data = []

        for ieta in available_etas:
            eta_label = self.ETA_LABELS_LATEX.get(ieta, f"$\\eta_{ieta}$")
            for m_key in method_keys:
                method_name = (
                    method_label_overrides.get(m_key)
                    if method_label_overrides is not None and m_key in method_label_overrides
                    else self.method_aliases.get(m_key, m_key)
                )
                index_tuples.append((eta_label, method_name))

                row_cells: List[str] = []
                for iet in available_ets:
                    sub_region = df[(df["iet"] == iet) & (df["ieta"] == ieta)]
                    best_sp_val = sub_region["sp_disp"].max() if not sub_region.empty else -1.0

                    row_match = sub_region[sub_region["method_key"] == m_key]
                    if not row_match.empty:
                        rec = row_match.iloc[0]
                        is_best = bold_best_sp and np.isclose(rec["sp_disp"], best_sp_val, atol=1e-5)
                        row_cells.append(
                            self._format_cell(
                                rec["pd_disp"],
                                rec["pd_err_disp"],
                                precision,
                                decimal_comma,
                                math_delimiters,
                                is_best,
                                allow_missing_err=allow_missing_err,
                            )
                        )
                        row_cells.append(
                            self._format_cell(
                                rec["sp_disp"],
                                rec["sp_err_disp"],
                                precision,
                                decimal_comma,
                                math_delimiters,
                                is_best,
                                allow_missing_err=allow_missing_err,
                            )
                        )
                        row_cells.append(
                            self._format_cell(
                                rec["pf_disp"],
                                rec["pf_err_disp"],
                                precision,
                                decimal_comma,
                                math_delimiters,
                                is_best,
                                allow_missing_err=allow_missing_err,
                            )
                        )
                    else:
                        row_cells.extend(["-", "-", "-"])
                rows_data.append(row_cells)

        multi_index = pd.MultiIndex.from_tuples(index_tuples, names=["eta", "Method"])
        return pd.DataFrame(rows_data, index=multi_index, columns=multi_cols)

    def build_grid_table(
        self,
        as_percentage: bool = True,
        precision: int = 2,
        decimal_comma: bool = False,
        math_delimiters: bool = True,
        bold_best_sp: bool = True,
    ) -> pd.DataFrame:
        """Constructs a 2D MultiIndex grid DataFrame formatted identically to the reference table.

        Args:
            as_percentage: Multiply metric values and uncertainties by 100.
            precision: Number of decimal places for formatted values.
            decimal_comma: Use comma as decimal separator.
            math_delimiters: Wrap expressions with LaTeX math delimiters.
            bold_best_sp: Bold the entry with the highest SP value within each region.

        Returns:
            DataFrame with MultiIndex columns (ET Interval, Metric) and MultiIndex rows (eta, Method).
        """
        return self._build_grid_from_dataframe(
            tidy_df=self.build_tidy_dataframe(),
            method_keys=list(self.DEFAULT_METHOD_KEYS),
            as_percentage=as_percentage,
            precision=precision,
            decimal_comma=decimal_comma,
            math_delimiters=math_delimiters,
            bold_best_sp=bold_best_sp,
            allow_missing_err=False,
        )

    def build_training_grid_table(
        self,
        as_percentage: bool = True,
        precision: int = 2,
        decimal_comma: bool = False,
        math_delimiters: bool = True,
        bold_best_sp: bool = False,
    ) -> pd.DataFrame:
        """Constructs a 2D MultiIndex grid DataFrame containing only the best-SP training metrics.

        Args:
            as_percentage: Multiply metric values by 100.
            precision: Number of decimal places for formatted values.
            decimal_comma: Use comma as decimal separator.
            math_delimiters: Wrap expressions with LaTeX math delimiters.
            bold_best_sp: Bold the entry in each region.

        Returns:
            DataFrame with MultiIndex columns (ET Interval, Metric) and MultiIndex rows (eta, Method).
        """
        return self._build_grid_from_dataframe(
            tidy_df=self.build_training_tidy_dataframe(),
            method_keys=[self.TRAINING_METHOD_KEY],
            as_percentage=as_percentage,
            precision=precision,
            decimal_comma=decimal_comma,
            math_delimiters=math_delimiters,
            bold_best_sp=bold_best_sp,
            allow_missing_err=False,
        )

    def build_athena_matched_comparison_grid_table(
        self,
        as_percentage: bool = True,
        precision: int = 2,
        decimal_comma: bool = False,
        math_delimiters: bool = True,
        bold_best_sp: bool = True,
        include_training_best_sp: bool = True,
    ) -> pd.DataFrame:
        """Constructs a multi-row-per-eta grid comparing Athena-matched cuts (optionally with Training Best SP).

        Args:
            as_percentage: Multiply metric values and uncertainties by 100.
            precision: Number of decimal places for formatted values.
            decimal_comma: Use comma as decimal separator.
            math_delimiters: Wrap expressions with LaTeX math delimiters.
            bold_best_sp: Bold the entry with the highest SP value within each region.
            include_training_best_sp: Whether to include the Training Best SP row.

        Returns:
            DataFrame with MultiIndex columns (ET Interval, Metric) and MultiIndex rows (eta, Method).
        """
        method_keys = (
            [self.TRAINING_METHOD_KEY, *self.DEFAULT_METHOD_KEYS]
            if include_training_best_sp
            else list(self.DEFAULT_METHOD_KEYS)
        )
        return self._build_grid_from_dataframe(
            tidy_df=self.build_athena_matched_tidy_dataframe(
                include_training_best_sp=include_training_best_sp
            ),
            method_keys=method_keys,
            as_percentage=as_percentage,
            precision=precision,
            decimal_comma=decimal_comma,
            math_delimiters=math_delimiters,
            bold_best_sp=bold_best_sp,
            allow_missing_err=False,
            method_label_overrides=self._athena_matched_method_aliases(),
        )

    @staticmethod
    def render_grid_to_latex(
        df: pd.DataFrame,
        caption: Optional[str] = None,
        label: Optional[str] = None,
        resize_box: bool = True,
    ) -> str:
        """Renders a pre-built 2D MultiIndex grid DataFrame into a LaTeX tabular snippet."""
        et_labels = list(dict.fromkeys(col[0] for col in df.columns))
        n_ets = len(et_labels)
        col_align = "|c|l|" + "|".join(["ccc"] * n_ets) + "|"

        lines: List[str] = [f"\\begin{{tabular}}{{{col_align}}}", "\\hline"]

        header_top = " & "
        for et in et_labels:
            header_top += f" & \\multicolumn{{3}}{{c|}}{{{et}}}"
        header_top += " \\\\"
        lines.append(header_top)

        header_sub = " & Method"
        for _ in range(n_ets):
            header_sub += " & $P_D[\\%]$ & $SP[\\%]$ & $P_F[\\%]$"
        header_sub += " \\\\"
        lines.append(header_sub)
        lines.append("\\hline")

        eta_groups = list(dict.fromkeys(df.index.get_level_values(0)))
        for eta_val in eta_groups:
            sub_df = df.loc[eta_val]
            n_methods = len(sub_df)

            for i, (method_name, row) in enumerate(sub_df.iterrows()):
                eta_cell = f"\\multirow{{{n_methods}}}{{*}}{{{eta_val}}}" if i == 0 else ""
                cells = [eta_cell, str(method_name)] + [str(c) for c in row.values]
                line_str = " & ".join(cells) + " \\\\"
                lines.append(line_str)
            lines.append("\\hline")

        lines.append("\\end{tabular}")
        return LatexTableFormatter.wrap_booktabs_environment(
            tabular_body="\n".join(lines),
            caption=caption,
            label=label,
            resize_to_textwidth=resize_box,
        )

    def to_latex_snippet(
        self,
        grid_table: Optional[pd.DataFrame] = None,
        caption: Optional[str] = None,
        label: Optional[str] = None,
        resize_box: bool = True,
        as_percentage: bool = True,
        precision: int = 2,
        decimal_comma: bool = False,
    ) -> str:
        """Converts consolidated operating point metrics into a LaTeX tabular snippet matching the image.

        Args:
            grid_table: Optional pre-built grid DataFrame.
            caption: Optional LaTeX table caption.
            label: Optional LaTeX reference label.
            resize_box: Wrap table inside resizebox for full textwidth fit.
            as_percentage: Scale metrics to percentages.
            precision: Decimal precision for numerical figures.
            decimal_comma: Use comma as decimal separator.

        Returns:
            Formatted LaTeX table block string.
        """
        df = grid_table if grid_table is not None else self.build_grid_table(
            as_percentage=as_percentage,
            precision=precision,
            decimal_comma=decimal_comma,
            math_delimiters=True,
            bold_best_sp=True,
        )
        return self.render_grid_to_latex(
            df=df,
            caption=caption,
            label=label,
            resize_box=resize_box,
        )

    def export_parquet(
        self,
        output_path: Union[str, Path],
        df: Optional[pd.DataFrame] = None,
    ) -> Path:
        """Serializes tidy or grid DataFrame to Parquet format.

        Args:
            output_path: Destination file path.
            df: Optional DataFrame to serialize (defaults to long-form tidy metrics).

        Returns:
            Resolved Path of the written parquet file.
        """
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        export_df = df if df is not None else self.build_tidy_dataframe()
        if isinstance(export_df.columns, pd.MultiIndex):
            flat_df = export_df.copy()
            flat_df.columns = [f"{c[0]}_{c[1]}" for c in export_df.columns]
            flat_df.to_parquet(path)
        else:
            export_df.to_parquet(path)
        return path

    def export_latex(
        self,
        output_path: Union[str, Path],
        latex_code: Optional[str] = None,
    ) -> Path:
        """Saves generated LaTeX table string to a .tex file.

        Args:
            output_path: Destination file path.
            latex_code: Optional LaTeX string.

        Returns:
            Resolved Path of the saved .tex file.
        """
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        content = latex_code if latex_code is not None else self.to_latex_snippet()
        path.write_text(content, encoding="utf-8")
        return path


@dataclass(frozen=True)
class ModelSpec:
    """Identifies a trained model configuration and its results directory."""

    label: str
    config_path: Union[str, Path]
    results_path: Union[str, Path]
    manifest_path: Optional[Union[str, Path]] = None

    @classmethod
    def from_results_dir(
        cls,
        results_path: Union[str, Path],
        config_dir: Union[str, Path] = Path("config/NeuralRinger"),
        label: Optional[str] = None,
        manifest_path: Optional[Union[str, Path]] = None,
    ) -> "ModelSpec":
        """Resolves a ModelSpec by parsing the config name from a results directory.

        Args:
            results_path: Path to a results folder named `config_name<Name>_id<Timestamp>`.
            config_dir: Directory containing model YAML configurations.
            label: Optional display label; defaults to the parsed config name.
            manifest_path: Optional split manifest path for this model.

        Returns:
            ModelSpec populated with the resolved config and results paths.
        """
        results_dir = Path(results_path)
        match = re.match(r"^config_name(.+)_id\d+$", results_dir.name)
        if not match:
            raise ValueError(
                f"Cannot parse config name from results folder '{results_dir.name}'. "
                "Expected pattern 'config_name<Name>_id<Timestamp>'."
            )
        config_name = match.group(1)
        resolved_config = Path(config_dir) / f"{config_name}.yaml"
        if not resolved_config.exists():
            raise FileNotFoundError(
                f"Resolved config file not found at '{resolved_config}' for results '{results_dir}'."
            )
        return cls(
            label=label or config_name,
            config_path=resolved_config,
            results_path=results_dir,
            manifest_path=manifest_path,
        )


class MultiModelAthenaMatchedConsolidator:
    """Consolidates Athena-matched Hard Cut and Linear Fit y_cut(<mu>) across multiple models."""

    def __init__(
        self,
        model_specs: Sequence[Union[ModelSpec, Tuple[str, Union[str, Path], Union[str, Path]]]],
        manifest_path: Optional[Union[str, Path]] = None,
        working_point: str = "loose",
        bins_mu: int = 10,
        n_bootstraps_pileup: int = 100,
        efficiencies_csv: Optional[Union[str, Path]] = None,
        device: Optional[Union[str, torch.device]] = None,
        include_training_best_sp: bool = True,
    ) -> None:
        """Initializes the multi-model Athena-matched consolidator.

        Args:
            model_specs: Sequence of ModelSpec or (label, config_path, results_path) tuples.
            manifest_path: Optional shared split manifest path.
            working_point: Athena fast photon cut working point ('loose', 'medium', 'tight').
            bins_mu: Bin count for pileup threshold calibration.
            n_bootstraps_pileup: Bootstrap sample count for pileup threshold fitting.
            efficiencies_csv: Optional path to reference efficiencies CSV.
            device: Computing device ('cpu', 'cuda', or torch.device).
            include_training_best_sp: Whether to include each model's Training Best SP row.
        """
        if not model_specs:
            raise ValueError("At least one model specification must be provided.")
        self.model_specs: List[ModelSpec] = [
            spec
            if isinstance(spec, ModelSpec)
            else ModelSpec(label=spec[0], config_path=spec[1], results_path=spec[2])
            for spec in model_specs
        ]
        self.manifest_path = manifest_path
        self.working_point = working_point.lower()
        self.bins_mu = bins_mu
        self.n_bootstraps_pileup = n_bootstraps_pileup
        self.efficiencies_csv = efficiencies_csv
        self.device = device
        self.include_training_best_sp = include_training_best_sp
        self._records: List[Dict[str, Any]] = []
        self._consolidators: List[RegionalOperatingPointsConsolidator] = []

    @classmethod
    def from_results_dirs(
        cls,
        results_dirs: Sequence[Union[str, Path]],
        config_dir: Union[str, Path] = Path("config/NeuralRinger"),
        labels: Optional[Sequence[str]] = None,
        **kwargs: Any,
    ) -> "MultiModelAthenaMatchedConsolidator":
        """Instantiates the consolidator directly from a list of results directories.

        Args:
            results_dirs: Sequence of results folders named `config_name<Name>_id<Timestamp>`.
            config_dir: Directory containing model YAML configurations.
            labels: Optional display labels aligned with `results_dirs`.
            **kwargs: Additional keyword arguments forwarded to `__init__`.

        Returns:
            Configured MultiModelAthenaMatchedConsolidator instance.
        """
        if labels is not None and len(labels) != len(results_dirs):
            raise ValueError("Length of `labels` must match length of `results_dirs`.")
        specs = [
            ModelSpec.from_results_dir(
                results_path=folder,
                config_dir=config_dir,
                label=labels[idx] if labels is not None else None,
            )
            for idx, folder in enumerate(results_dirs)
        ]
        return cls(model_specs=specs, **kwargs)

    def _method_keys_and_labels(self) -> Tuple[List[str], Dict[str, str]]:
        wp_label = self.working_point.capitalize()
        keys: List[str] = ["athena_cut"]
        labels: Dict[str, str] = {"athena_cut": f"Athena {wp_label} Cut"}
        for idx, spec in enumerate(self.model_specs):
            if self.include_training_best_sp:
                best_sp_key = f"model_{idx}_best_sp"
                keys.append(best_sp_key)
                labels[best_sp_key] = f"{spec.label} [Best SP]"
            hard_key = f"model_{idx}_hard_cut"
            pileup_key = f"model_{idx}_pileup_cut"
            keys.extend([hard_key, pileup_key])
            labels[hard_key] = f"{spec.label} [Hard Cut]"
            labels[pileup_key] = f"{spec.label} [$y_\\mathrm{{cut}}(\\langle\\mu\\rangle)$]"
        return keys, labels

    def scan_and_evaluate(self) -> List[Dict[str, Any]]:
        """Evaluates Hard Cut and Pileup-Corrected Cut at Athena PD (and optionally Best SP) for all models."""
        self._records = []
        self._consolidators = []
        seen_athena_regions: set[Tuple[int, int]] = set()
        _, labels = self._method_keys_and_labels()

        for idx, spec in enumerate(self.model_specs):
            consolidator = RegionalOperatingPointsConsolidator(
                config_path=spec.config_path,
                results_path=spec.results_path,
                manifest_path=spec.manifest_path or self.manifest_path,
                working_point=self.working_point,
                bins_mu=self.bins_mu,
                n_bootstraps_pileup=self.n_bootstraps_pileup,
                efficiencies_csv=self.efficiencies_csv,
                device=self.device,
            )
            self._consolidators.append(consolidator)
            model_records = consolidator.scan_athena_matched_and_training(
                include_training_best_sp=self.include_training_best_sp
            )

            for rec in model_records:
                region = (int(rec["iet"]), int(rec["ieta"]))
                src_key = rec["method_key"]
                if src_key == "athena_cut":
                    if region in seen_athena_regions:
                        continue
                    seen_athena_regions.add(region)
                    mapped_key = "athena_cut"
                    model_label = "Athena"
                elif src_key == RegionalOperatingPointsConsolidator.TRAINING_METHOD_KEY:
                    mapped_key = f"model_{idx}_best_sp"
                    model_label = spec.label
                elif src_key == "calibrated_hard_cut":
                    mapped_key = f"model_{idx}_hard_cut"
                    model_label = spec.label
                elif src_key == "pileup_corrected_cut":
                    mapped_key = f"model_{idx}_pileup_cut"
                    model_label = spec.label
                else:
                    continue

                cfg_obj = getattr(consolidator, "config", None)
                config_name = "Athena" if mapped_key == "athena_cut" else getattr(cfg_obj, "config_name", spec.label)
                self._records.append(
                    {
                        **rec,
                        "method_key": mapped_key,
                        "method": labels[mapped_key],
                        "model_label": model_label,
                        "config_name": config_name,
                    }
                )

        log.info(
            f"Multi-model Athena-matched evaluation complete. Extracted {len(self._records)} records."
        )
        return self._records

    def build_tidy_dataframe(self) -> pd.DataFrame:
        """Returns the consolidated multi-model metrics as a tidy DataFrame."""
        if not self._records:
            self.scan_and_evaluate()
        return pd.DataFrame(self._records)

    def build_grid_table(
        self,
        as_percentage: bool = True,
        precision: int = 2,
        decimal_comma: bool = False,
        math_delimiters: bool = True,
        bold_best_sp: bool = True,
    ) -> pd.DataFrame:
        """Builds a 2D MultiIndex grid comparing Athena Cut and each model's Hard/Pileup cuts."""
        tidy_df = self.build_tidy_dataframe()
        method_keys, method_labels = self._method_keys_and_labels()
        helper = self._consolidators[0] if self._consolidators else RegionalOperatingPointsConsolidator.__new__(
            RegionalOperatingPointsConsolidator
        )
        if not hasattr(helper, "method_aliases"):
            helper.method_aliases = method_labels
        return helper._build_grid_from_dataframe(
            tidy_df=tidy_df,
            method_keys=method_keys,
            as_percentage=as_percentage,
            precision=precision,
            decimal_comma=decimal_comma,
            math_delimiters=math_delimiters,
            bold_best_sp=bold_best_sp,
            allow_missing_err=False,
            method_label_overrides=method_labels,
        )

    def to_latex_snippet(
        self,
        grid_table: Optional[pd.DataFrame] = None,
        caption: Optional[str] = None,
        label: Optional[str] = None,
        resize_box: bool = True,
        as_percentage: bool = True,
        precision: int = 2,
        decimal_comma: bool = False,
    ) -> str:
        """Converts the multi-model grid table into a publication-ready LaTeX snippet."""
        df = (
            grid_table
            if grid_table is not None
            else self.build_grid_table(
                as_percentage=as_percentage,
                precision=precision,
                decimal_comma=decimal_comma,
                math_delimiters=True,
                bold_best_sp=True,
            )
        )
        return RegionalOperatingPointsConsolidator.render_grid_to_latex(
            df=df,
            caption=caption,
            label=label,
            resize_box=resize_box,
        )

    def export_parquet(
        self,
        output_path: Union[str, Path],
        df: Optional[pd.DataFrame] = None,
    ) -> Path:
        """Serializes tidy or grid DataFrame to Parquet format."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        export_df = df if df is not None else self.build_tidy_dataframe()
        if isinstance(export_df.columns, pd.MultiIndex):
            flat_df = export_df.copy()
            flat_df.columns = [f"{c[0]}_{c[1]}" for c in export_df.columns]
            flat_df.to_parquet(path)
        else:
            export_df.to_parquet(path)
        return path

    def export_latex(
        self,
        output_path: Union[str, Path],
        latex_code: Optional[str] = None,
    ) -> Path:
        """Saves generated LaTeX table string to a .tex file."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        content = latex_code if latex_code is not None else self.to_latex_snippet()
        path.write_text(content, encoding="utf-8")
        return path


