import gc
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import torch

from src.core.Datasets.SplitManifest import SplitManifest
from src.core.Trainers.NeuralRingerTrainer import NeuralRingerTrainer
from src.core.Validation.FastPhotonCutEvaluator import (
    FastPhotonCutEvaluator,
    UserKinematicGrid,
)
from src.core.Validation.HoldoutEvaluator import HoldoutEvaluator
from src.core.Validation.PileupAnalyzer import PileupAnalyzer
from src.core.Validation.PileupThresholdCalibrator import PileupThresholdCalibrator
from src.core.Validation.QuadrantAnalyzer import StrategyMetrics
from src.core.Validation.ResultAggregator import ResultAggregator
from src.Parser.DynamicConfiguration import DynamicConfiguration
from src.Parser.NeuralRingerTrainerConfiguration import NeuralRingerTrainerConfiguration
from src.utils import get_et_eta

log = logging.getLogger("RegionalOperatingPointsConsolidator")


class RegionalOperatingPointsConsolidator:
    """Consolidates operating point performance metrics across all kinematic regions."""

    ET_LABELS_LATEX: Dict[int, str] = {
        0: "$E_{T0}$",
        1: "$E_{T1}$",
        2: "$E_{T2}$",
        3: "$E_{T3}$",
        4: "$E_{T4}$",
    }

    ETA_LABELS_LATEX: Dict[int, str] = {
        0: "$\\eta_0$",
        1: "$\\eta_1$",
        2: "$\\eta_2$",
        3: "$\\eta_3$",
        4: "$\\eta_4$",
    }

    DEFAULT_METHOD_KEYS: Tuple[str, str, str] = (
        "calibrated_hard_cut",
        "pileup_corrected_cut",
        "athena_cut",
    )

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

        self.config: NeuralRingerTrainerConfiguration = DynamicConfiguration.from_yaml(
            str(self.config_path), NeuralRingerTrainerConfiguration
        )
        self.trainer: NeuralRingerTrainer = NeuralRingerTrainer(self.config)

        resolved_manifest = manifest_path or (self.results_path.parent / "split_manifest.json")
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
        }
        if method_aliases:
            self.method_aliases.update(method_aliases)

        self._records: List[Dict[str, Any]] = []

    @property
    def records(self) -> List[Dict[str, Any]]:
        """Returns the list of extracted regional evaluation records."""
        return self._records

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
                best_model, best_rep, mean_sp, std_sp = self.aggregator.get_best_model_for_region(iet, ieta)
                model.load_state_dict(best_model["best_weights"])
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

            best_fold = int(best_model.get("fold", 0))
            train_idx, val_idx = train_folds[best_fold]
            calib_idx = np.concatenate([train_idx, val_idx])

            calib_features = raw_features[calib_idx]
            calib_labels = raw_labels[calib_idx]
            calib_ring_features = calib_features[:, ring_indices]

            logits_calib = HoldoutEvaluator.predict_logits(model, calib_ring_features, self.device)
            mu_calib = PileupAnalyzer.extract_feature(calib_features, feature_names_list, "avgmu")
            mu_test = PileupAnalyzer.extract_feature(eval_res.holdout_features_full, feature_names_list, "avgmu")

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
                    {
                        "iet": iet,
                        "ieta": ieta,
                        "method_key": key,
                        "method": name,
                        "target_pd": target_pd,
                        "decision_threshold": th_repr,
                        "pd": met.pd,
                        "pd_uncertainty": met.pd_uncertainty,
                        "sp": met.sp,
                        "sp_uncertainty": met.sp_uncertainty,
                        "pf": met.pf,
                        "pf_uncertainty": met.pf_uncertainty,
                        "eff": met.eff,
                        "eff_uncertainty": met.eff_uncertainty,
                        "bg_eff": met.bg_eff,
                        "bg_eff_uncertainty": met.bg_eff_uncertainty,
                        "tp": met.tp,
                        "fp": met.fp,
                        "tn": met.tn,
                        "fn": met.fn,
                        "total": met.total,
                    }
                )

            log.info(
                f"[iet{iet}.ieta{ieta}] HardCut(SP={metrics_hard.sp:.4f}) | "
                f"PileupCut(SP={metrics_pileup.sp:.4f}) | Athena(SP={metrics_athena.sp:.4f})"
            )

            del raw_features, raw_labels, eval_res, calib_features, calib_labels
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        log.info(f"Evaluation complete. Extracted {len(self._records)} method records.")
        return self._records

    def build_tidy_dataframe(self) -> pd.DataFrame:
        """Returns the extracted metrics in long-form tidy DataFrame format."""
        if not self._records:
            self.scan_and_evaluate()
        return pd.DataFrame(self._records)

    def _format_cell(
        self,
        value: float,
        uncertainty: float,
        precision: int = 2,
        decimal_comma: bool = False,
        math_delimiters: bool = True,
        bold: bool = False,
    ) -> str:
        if np.isnan(value) or np.isnan(uncertainty):
            return "-"
        val_str = f"{value:.{precision}f}"
        err_str = f"{uncertainty:.{precision}f}"
        if decimal_comma:
            val_str = val_str.replace(".", "{,}")
            err_str = err_str.replace(".", "{,}")
        core = f"{val_str} \\pm {err_str}"
        if bold:
            core = f"\\mathbf{{{val_str} \\pm {err_str}}}"
        return f"${core}$" if math_delimiters else core

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
        tidy_df = self.build_tidy_dataframe().copy()
        if tidy_df.empty:
            raise RuntimeError("No regional evaluation records available.")

        scale = 100.0 if as_percentage else 1.0
        tidy_df["pd_disp"] = tidy_df["pd"] * scale
        tidy_df["pd_err_disp"] = tidy_df["pd_uncertainty"] * scale
        tidy_df["sp_disp"] = tidy_df["sp"] * scale
        tidy_df["sp_err_disp"] = tidy_df["sp_uncertainty"] * scale
        tidy_df["pf_disp"] = tidy_df["pf"] * scale
        tidy_df["pf_err_disp"] = tidy_df["pf_uncertainty"] * scale

        available_ets = sorted(tidy_df["iet"].unique().astype(int))
        available_etas = sorted(tidy_df["ieta"].unique().astype(int))
        method_keys = list(self.DEFAULT_METHOD_KEYS)

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
                method_name = self.method_aliases[m_key]
                index_tuples.append((eta_label, method_name))

                row_cells: List[str] = []
                for iet in available_ets:
                    sub_region = tidy_df[(tidy_df["iet"] == iet) & (tidy_df["ieta"] == ieta)]
                    best_sp_val = sub_region["sp_disp"].max() if not sub_region.empty else -1.0

                    row_match = sub_region[sub_region["method_key"] == m_key]
                    if not row_match.empty:
                        rec = row_match.iloc[0]
                        is_best = bold_best_sp and np.isclose(rec["sp_disp"], best_sp_val, atol=1e-5)
                        row_cells.append(
                            self._format_cell(
                                rec["pd_disp"], rec["pd_err_disp"], precision, decimal_comma, math_delimiters, is_best
                            )
                        )
                        row_cells.append(
                            self._format_cell(
                                rec["sp_disp"], rec["sp_err_disp"], precision, decimal_comma, math_delimiters, is_best
                            )
                        )
                        row_cells.append(
                            self._format_cell(
                                rec["pf_disp"], rec["pf_err_disp"], precision, decimal_comma, math_delimiters, is_best
                            )
                        )
                    else:
                        row_cells.extend(["-", "-", "-"])
                rows_data.append(row_cells)

        multi_index = pd.MultiIndex.from_tuples(index_tuples, names=["eta", "Método"])
        return pd.DataFrame(rows_data, index=multi_index, columns=multi_cols)

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
        df = grid_table or self.build_grid_table(
            as_percentage=as_percentage,
            precision=precision,
            decimal_comma=decimal_comma,
            math_delimiters=True,
            bold_best_sp=True,
        )

        et_labels = sorted(list(set(col[0] for col in df.columns)), key=lambda x: str(x))
        n_ets = len(et_labels)
        total_cols = 2 + 3 * n_ets
        col_align = "|c|l|" + "|".join(["ccc"] * n_ets) + "|"

        lines: List[str] = []
        lines.append("\\begin{table}[htbp]")
        lines.append("\\centering")
        if caption:
            lines.append(f"\\caption{{{caption}}}")
        if label:
            lines.append(f"\\label{{{label}}}")

        if resize_box:
            lines.append("\\resizebox{\\textwidth}{!}{%")

        lines.append(f"\\begin{{tabular}}{{{col_align}}}")
        lines.append("\\hline")

        header_top = " & "
        for et in et_labels:
            header_top += f" & \\multicolumn{{3}}{{c|}}{{{et}}}"
        header_top += " \\\\"
        lines.append(header_top)

        header_sub = " & Método"
        for _ in range(n_ets):
            header_sub += " & $P_D[\\%]$ & $SP[\\%]$ & $P_F[\\%]$"
        header_sub += " \\\\"
        lines.append(header_sub)
        lines.append("\\hline")

        eta_groups = df.index.levels[0]
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
        if resize_box:
            lines.append("}")
        lines.append("\\end{table}")
        return "\n".join(lines) + "\n"

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

