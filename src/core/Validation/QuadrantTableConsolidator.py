import pathlib
from typing import Optional, Union
import numpy as np
import pandas as pd


class QuadrantTableConsolidator:
    """Consolidates regional quadrant analysis results into formatted summary tables."""

    def __init__(self, csv_source: Union[str, pathlib.Path, pd.DataFrame]) -> None:
        """Initializes the consolidator with a CSV path or existing DataFrame.

        Args:
            csv_source: Path to quadrant results CSV or pre-loaded DataFrame.
        """
        if isinstance(csv_source, pd.DataFrame):
            self._df = csv_source.copy()
        else:
            self._df = pd.read_csv(pathlib.Path(csv_source))
        self._df.columns = self._df.columns.str.strip()

    @property
    def raw_data(self) -> pd.DataFrame:
        """Returns the raw input DataFrame."""
        return self._df

    def get_strategy_name(self, strategy: str = "strategy1") -> str:
        """Returns the configured strategy name from the table records.

        Args:
            strategy: Target strategy tag ('strategy1' or 'strategy2').

        Returns:
            Name identifier of the requested strategy.
        """
        col_name = "strategy1_name" if strategy.lower() == "strategy1" else "strategy2_name"
        if col_name in self._df.columns and len(self._df) > 0:
            return str(self._df[col_name].dropna().iloc[0])
        return strategy

    def build_numeric_table(
        self,
        strategy: str = "strategy1",
        include_global: bool = True,
    ) -> pd.DataFrame:
        """Extracts regional records with numeric values and uncertainties.

        Args:
            strategy: Target strategy ('strategy1' or 'strategy2').
            include_global: Whether to append the global composite row.

        Returns:
            DataFrame containing extracted numeric metrics for each region.
        """
        prefix = "s1_" if strategy.lower() == "strategy1" else "s2_"
        df = self._df.copy()

        pd_col = f"{prefix}pd"
        pf_col = f"{prefix}pf"
        sp_col = f"{prefix}sp"

        pd_err_col = (
            f"{prefix}sig_eff_uncertainty"
            if f"{prefix}sig_eff_uncertainty" in df.columns
            else f"{prefix}pd_uncertainty"
        )
        pf_err_col = (
            f"{prefix}bg_eff_uncertainty"
            if f"{prefix}bg_eff_uncertainty" in df.columns
            else f"{prefix}pf_uncertainty"
        )
        sp_err_col = f"{prefix}sp_uncertainty"

        is_global = (df["region"].astype(str).str.upper() == "GLOBAL") | (df["iet"] < 0)
        regional_df = df[~is_global].sort_values(by=["iet", "ieta"]).copy()

        rows = []
        for _, row in regional_df.iterrows():
            rows.append(self._extract_row_record(row, pd_col, pd_err_col, pf_col, pf_err_col, sp_col, sp_err_col))

        if include_global:
            global_df = df[is_global]
            if len(global_df) > 0:
                rows.append(self._extract_row_record(global_df.iloc[0], pd_col, pd_err_col, pf_col, pf_err_col, sp_col, sp_err_col))

        return pd.DataFrame(rows)

    def _extract_row_record(
        self,
        row: pd.Series,
        pd_col: str,
        pd_err_col: str,
        pf_col: str,
        pf_err_col: str,
        sp_col: str,
        sp_err_col: str,
    ) -> dict:
        iet = int(row["iet"]) if not pd.isna(row["iet"]) else -1
        ieta = int(row["ieta"]) if not pd.isna(row["ieta"]) else -1
        region = str(row["region"]).strip()

        sp_val = float(row[sp_col]) if sp_col in row and not pd.isna(row[sp_col]) else np.nan
        sp_err = float(row[sp_err_col]) if sp_err_col in row and not pd.isna(row[sp_err_col]) else np.nan

        pd_val = float(row[pd_col]) if pd_col in row and not pd.isna(row[pd_col]) else np.nan
        pd_err = float(row[pd_err_col]) if pd_err_col in row and not pd.isna(row[pd_err_col]) else np.nan

        pf_val = float(row[pf_col]) if pf_col in row and not pd.isna(row[pf_col]) else np.nan
        pf_err = float(row[pf_err_col]) if pf_err_col in row and not pd.isna(row[pf_err_col]) else np.nan

        return {
            "region": region,
            "iet": iet,
            "ieta": ieta,
            "sp": sp_val,
            "sp_uncertainty": sp_err,
            "pd": pd_val,
            "pd_uncertainty": pd_err,
            "pf": pf_val,
            "pf_uncertainty": pf_err,
        }

    @staticmethod
    def _format_cell(
        val: float,
        err: float,
        precision: int = 4,
        math_delimiters: bool = True,
    ) -> str:
        if pd.isna(val):
            return "-"

        if pd.isna(err):
            content = f"{val:.{precision}f}"
        else:
            content = f"{val:.{precision}f} \\pm {err:.{precision}f}"

        return f"${content}$" if math_delimiters else content

    def build_latex_formatted_table(
        self,
        strategy: str = "strategy1",
        math_delimiters: bool = True,
        precision: int = 4,
        include_global: bool = True,
    ) -> pd.DataFrame:
        """Constructs 25-region summary table with LaTeX-formatted string cells.

        Args:
            strategy: Target strategy ('strategy1' or 'strategy2').
            math_delimiters: Enclose cell strings with dollar signs.
            precision: Decimal precision for numerical figures.
            include_global: Whether to append the global composite row.

        Returns:
            DataFrame with formatted strings for SP, PD, and PF.
        """
        numeric_df = self.build_numeric_table(strategy=strategy, include_global=include_global)

        formatted_records = []
        for _, row in numeric_df.iterrows():
            formatted_records.append(
                {
                    "Region": row["region"],
                    "iet": row["iet"],
                    "ieta": row["ieta"],
                    "SP": self._format_cell(row["sp"], row["sp_uncertainty"], precision, math_delimiters),
                    "PD": self._format_cell(row["pd"], row["pd_uncertainty"], precision, math_delimiters),
                    "PF": self._format_cell(row["pf"], row["pf_uncertainty"], precision, math_delimiters),
                }
            )

        return pd.DataFrame(formatted_records)

    def export_csv(
        self,
        df: pd.DataFrame,
        output_path: Union[str, pathlib.Path],
    ) -> pathlib.Path:
        """Saves tabular data to a CSV file.

        Args:
            df: DataFrame to serialize.
            output_path: Destination filepath.

        Returns:
            Path of the saved file.
        """
        path = pathlib.Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        return path

    def to_latex_snippet(
        self,
        df: pd.DataFrame,
        caption: Optional[str] = None,
        label: Optional[str] = None,
    ) -> str:
        """Converts formatted DataFrame to a LaTeX tabular snippet.

        Args:
            df: Formatted DataFrame.
            caption: Optional table caption.
            label: Optional table reference label.

        Returns:
            String containing the formatted LaTeX code block.
        """
        display_df = df.copy()
        if "iet" in display_df.columns and "ieta" in display_df.columns:
            display_df["Region"] = display_df.apply(
                lambda r: "GLOBAL" if r["iet"] < 0 else f"$E_T={r['iet']}, \\eta={r['ieta']}$",
                axis=1,
            )
            display_df = display_df.drop(columns=["iet", "ieta"])

        latex_body = display_df.to_latex(
            index=False,
            escape=False,
            column_format="l c c c",
        )

        caption_block = f"\\caption{{{caption}}}\n" if caption else ""
        label_block = f"\\label{{{label}}}\n" if label else ""

        return (
            "\\begin{table}[htbp]\n"
            "\\centering\n"
            f"{caption_block}"
            f"{label_block}"
            f"{latex_body}"
            "\\end{table}\n"
        )
