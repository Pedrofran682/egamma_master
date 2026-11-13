import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import pandas as pd

from src.core.Validation.LatexTableFormatter import LatexTableFormatter


class DistanceMetricsETL:
    """ETL processor for distance metric CSV tables generated during validation."""

    def __init__(
        self,
        folder_paths: Union[str, Path, List[Union[str, Path]]],
        file_pattern: str = "*distance_metrics_*.csv",
    ) -> None:
        """Initializes the ETL with one or multiple folder paths and file pattern.

        Args:
            folder_paths: Directory path or list of directory paths to scan.
            file_pattern: Filename glob pattern to match distance metric files.
        """
        if isinstance(folder_paths, (str, Path)):
            self._folder_paths = [Path(folder_paths)]
        else:
            self._folder_paths = [Path(p) for p in folder_paths]

        self._file_pattern = file_pattern
        self._discovered_files: List[Path] = self._discover_files()

    @property
    def folder_paths(self) -> List[Path]:
        return self._folder_paths

    @property
    def file_pattern(self) -> str:
        return self._file_pattern

    @property
    def discovered_files(self) -> List[Path]:
        return self._discovered_files

    def _discover_files(self) -> List[Path]:
        matched_files: List[Path] = []
        for folder in self._folder_paths:
            if not folder.exists():
                continue

            target_dirs = []
            tables_dir = folder / "Tables"
            if tables_dir.is_dir():
                target_dirs.append(tables_dir)
            elif folder.name == "Tables":
                target_dirs.append(folder)
            else:
                target_dirs.append(folder)
                sub_tables = list(folder.glob("**/Tables"))
                target_dirs.extend([d for d in sub_tables if d.is_dir()])

            for d in set(target_dirs):
                matched_files.extend(sorted(d.glob(self._file_pattern)))

        seen = set()
        unique_files = []
        for f in matched_files:
            resolved = f.resolve()
            if resolved not in seen:
                seen.add(resolved)
                unique_files.append(f)

        return unique_files

    @staticmethod
    def extract_et_eta(file_path: Union[str, Path]) -> Tuple[int, int]:
        """Extracts integer (ET, ETA) indices from filename.

        Args:
            file_path: Path or filename containing 'et' and 'eta' identifiers.

        Returns:
            Tuple of (et, eta) integers.

        Raises:
            ValueError: If pattern cannot be found.
        """
        name = Path(file_path).name
        match = re.search(r"et_?(\d+).*?eta_?(\d+)", name, re.IGNORECASE)
        if match:
            return int(match.group(1)), int(match.group(2))
        raise ValueError(f"Could not parse ET and ETA from filename: {name}")

    @staticmethod
    def extract_execution_folder(file_path: Union[str, Path]) -> str:
        """Extracts the parent execution folder of the 'Tables' directory.

        Args:
            file_path: Path to the target CSV file.

        Returns:
            Name of the execution directory containing 'Tables', or immediate parent.
        """
        path = Path(file_path).resolve()
        if path.parent.name == "Tables":
            return path.parent.parent.name
        for parent in path.parents:
            if (parent / "Tables").is_dir():
                return parent.name
        return path.parent.name

    @staticmethod
    def extract_metric_type(file_path: Union[str, Path]) -> str:
        """Extracts the distance metric comparison descriptor from filename."""
        name = Path(file_path).stem
        match = re.search(r"distance_metrics_(.+)$", name)
        if match:
            return match.group(1)
        return name

    def load_single_csv(self, file_path: Union[str, Path]) -> pd.DataFrame:
        """Loads a single CSV, parsing and appending ET, ETA, ExecutionFolder, and FolderPath columns.

        Args:
            file_path: Path to the CSV file.

        Returns:
            DataFrame with enriched columns.
        """
        path = Path(file_path)
        df = pd.read_csv(path)
        et, eta = self.extract_et_eta(path)
        execution_folder = self.extract_execution_folder(path)
        metric_type = self.extract_metric_type(path)

        df["ET"] = et
        df["ETA"] = eta
        df["ExecutionFolder"] = execution_folder
        df["FolderPath"] = str(path.parent)
        df["MetricType"] = metric_type
        return df

    def extract_and_consolidate(self) -> pd.DataFrame:
        """Loads and consolidates all discovered CSV files into a unified DataFrame.

        Returns:
            Consolidated DataFrame with all records and added columns.
        """
        if not self._discovered_files:
            return pd.DataFrame(columns=["ET", "ETA", "ExecutionFolder", "FolderPath", "MetricType"])

        dfs = [self.load_single_csv(f) for f in self._discovered_files]
        consolidated = pd.concat(dfs, ignore_index=True)
        return consolidated

    def get_summary(self, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Generates a summary of loaded records grouped by ExecutionFolder, FolderPath, ET, and ETA."""
        target_df = df if df is not None else self.extract_and_consolidate()
        if target_df.empty:
            return pd.DataFrame()

        group_cols = (
            ["ExecutionFolder", "FolderPath", "ET", "ETA", "MetricType"]
            if "FolderPath" in target_df.columns
            else ["ExecutionFolder", "ET", "ETA", "MetricType"]
        )
        summary = (
            target_df.groupby(group_cols)
            .size()
            .reset_index(name="RecordCount")
        )
        return summary

    @staticmethod
    def evaluate_distribution_equivalence(
        df: pd.DataFrame,
        alpha_p_value: float = 0.05,
        ks_effect_threshold: float = 0.02,
        kl_threshold: float = 0.01,
        js_threshold: float = 0.01,
        sigma_factor: float = 2.0,
    ) -> pd.DataFrame:
        """Evaluates equivalence for KS, Wasserstein, KL, and JS metric pairs.

        Args:
            df: DataFrame containing statistical distance metrics and uncertainties.
            alpha_p_value: Significance level for KS test p-value.
            ks_effect_threshold: Maximum KS distance for functional equivalence.
            kl_threshold: Maximum KL divergence for functional equivalence.
            js_threshold: Maximum JS divergence for functional equivalence.
            sigma_factor: Multiplier for bootstrap uncertainties.

        Returns:
            DataFrame copy enriched with boolean equivalence columns for each metric.
        """
        enriched = df.copy()

        if "KS Distance (D)" in enriched.columns and "KS Uncertainty" in enriched.columns:
            p_val = enriched["p-value"] if "p-value" in enriched.columns else 0.0
            ks_stat = enriched["KS Distance (D)"]
            ks_unc = enriched["KS Uncertainty"]
            enriched["KS_Equivalent"] = (p_val >= alpha_p_value) | (
                (ks_stat <= sigma_factor * ks_unc) & (ks_stat < ks_effect_threshold)
            )

        if "Wasserstein Distance" in enriched.columns and "Wasserstein Uncertainty" in enriched.columns:
            w_dist = enriched["Wasserstein Distance"]
            w_unc = enriched["Wasserstein Uncertainty"]
            enriched["Wasserstein_Equivalent"] = w_dist <= sigma_factor * w_unc

        if "KL Divergence" in enriched.columns and "KL Uncertainty" in enriched.columns:
            kl_div = enriched["KL Divergence"]
            kl_unc = enriched["KL Uncertainty"]
            enriched["KL_Equivalent"] = (kl_div <= sigma_factor * kl_unc) | (kl_div < kl_threshold)

        if "JS Divergence" in enriched.columns and "JS Uncertainty" in enriched.columns:
            js_div = enriched["JS Divergence"]
            js_unc = enriched["JS Uncertainty"]
            enriched["JS_Equivalent"] = (js_div <= sigma_factor * js_unc) | (js_div < js_threshold)

        return enriched

    @staticmethod
    def format_table_with_uncertainties(
        df: pd.DataFrame,
        pairs: Sequence[Union[Tuple[str, str], Tuple[str, str, str]]],
        id_cols: Optional[List[str]] = None,
        precision: int = 4,
        column_format: Optional[str] = None,
    ) -> Tuple[pd.DataFrame, str]:
        """Formats arbitrary metric values and uncertainty column pairs into LaTeX representation.

        Args:
            df: Source DataFrame containing values and uncertainty metrics.
            pairs: Sequence of either (display_name, value_col, unc_col) or (value_col, unc_col).
            id_cols: Optional identifier columns to preserve and escape in output.
            precision: Decimal precision for floating point numbers.
            column_format: Optional LaTeX column formatting string (e.g., 'l l c c').

        Returns:
            Tuple of (formatted_dataframe, latex_table_string).
        """
        formatted = pd.DataFrame()

        if id_cols is not None:
            for col in id_cols:
                if col in df.columns:
                    formatted[col] = df[col].apply(
                        lambda s: LatexTableFormatter.escape_math_aware(str(s))
                    )

        norm_specs: List[Tuple[str, str, str]] = []
        for item in pairs:
            if len(item) == 3:
                norm_specs.append((item[0], item[1], item[2]))
            elif len(item) == 2:
                norm_specs.append((item[0], item[0], item[1]))
            else:
                raise ValueError(f"Each pair must be (value_col, unc_col) or (display_name, value_col, unc_col), got: {item}")

        for col_name, val_col, unc_col in norm_specs:
            if val_col in df.columns and unc_col in df.columns:
                formatted[col_name] = [
                    LatexTableFormatter.format_value_with_uncertainty(
                        val, unc, precision=precision
                    )
                    for val, unc in zip(df[val_col], df[unc_col])
                ]

        if column_format is None:
            n_id = len([c for c in (id_cols or []) if c in df.columns])
            n_vals = sum(1 for _, v, u in norm_specs if v in df.columns and u in df.columns)
            column_format = " ".join(["l"] * n_id + ["c"] * n_vals)

        latex_str = formatted.to_latex(index=False, escape=False, column_format=column_format)
        return formatted, latex_str

    @staticmethod
    def format_latex_table(
        df: pd.DataFrame,
        precision: int = 4,
        features: Optional[List[str]] = None,
        comparisons: Optional[List[str]] = None,
        id_cols: Optional[List[str]] = None,
    ) -> Tuple[pd.DataFrame, str]:
        """Formats distance metrics and uncertainties into a LaTeX table representation (Legacy).

        Args:
            df: DataFrame containing Feature, Comparison, and metric/uncertainty pairs.
            precision: Decimal precision for floating point numbers.
            features: Optional filter for specific feature names.
            comparisons: Optional filter for specific comparison strings.
            id_cols: Optional identifier columns to preserve (defaults to ['Feature', 'Comparison']).

        Returns:
            Tuple of (formatted_dataframe, latex_table_string).
        """
        filtered = df.copy()
        if features is not None:
            filtered = filtered[filtered["Feature"].isin(features)]
        if comparisons is not None:
            filtered = filtered[filtered["Comparison"].isin(comparisons)]

        metric_specs = [
            ("KS Distance ($D$)", "KS Distance (D)", "KS Uncertainty"),
            ("Wasserstein ($W_1$)", "Wasserstein Distance", "Wasserstein Uncertainty"),
            ("KL Divergence", "KL Divergence", "KL Uncertainty"),
            ("JS Divergence", "JS Divergence", "JS Uncertainty"),
        ]

        return DistanceMetricsETL.format_table_with_uncertainties(
            df=filtered,
            pairs=metric_specs,
            id_cols=id_cols if id_cols is not None else ["Feature", "Comparison"],
            precision=precision,
            column_format="l l c c c c" if id_cols is None else None,
        )


