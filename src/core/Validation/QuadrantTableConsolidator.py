import pathlib
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd


class QuadrantTableConsolidator:
    """Consolidates regional quadrant analysis results into formatted summary tables."""

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

    ET_LABELS_PLAIN: Dict[int, str] = {
        0: "ET [15, 20) GeV",
        1: "ET [20, 30) GeV",
        2: "ET [30, 40) GeV",
        3: "ET [40, 50) GeV",
        4: "ET >= 50 GeV",
    }

    ETA_LABELS_PLAIN: Dict[int, str] = {
        0: "eta [0.00, 0.80)",
        1: "eta [0.80, 1.37)",
        2: "eta [1.37, 1.54)",
        3: "eta [1.54, 2.37)",
        4: "eta [2.37, 2.50)",
    }

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
        as_percentage: bool = False,
        include_global: bool = True,
    ) -> pd.DataFrame:
        """Extracts regional records with numeric values and uncertainties.

        Args:
            strategy: Target strategy ('strategy1' or 'strategy2').
            as_percentage: Multiply values and uncertainties by 100.
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

        scale = 100.0 if as_percentage else 1.0

        rows = []
        for _, row in regional_df.iterrows():
            rows.append(self._extract_row_record(row, pd_col, pd_err_col, pf_col, pf_err_col, sp_col, sp_err_col, scale))

        if include_global:
            global_df = df[is_global]
            if len(global_df) > 0:
                rows.append(self._extract_row_record(global_df.iloc[0], pd_col, pd_err_col, pf_col, pf_err_col, sp_col, sp_err_col, scale))

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
        scale: float,
    ) -> dict:
        iet = int(row["iet"]) if not pd.isna(row["iet"]) else -1
        ieta = int(row["ieta"]) if not pd.isna(row["ieta"]) else -1
        region = str(row["region"]).strip()

        sp_val = float(row[sp_col]) * scale if sp_col in row and not pd.isna(row[sp_col]) else np.nan
        sp_err = float(row[sp_err_col]) * scale if sp_err_col in row and not pd.isna(row[sp_err_col]) else np.nan

        pd_val = float(row[pd_col]) * scale if pd_col in row and not pd.isna(row[pd_col]) else np.nan
        pd_err = float(row[pd_err_col]) * scale if pd_err_col in row and not pd.isna(row[pd_err_col]) else np.nan

        pf_val = float(row[pf_col]) * scale if pf_col in row and not pd.isna(row[pf_col]) else np.nan
        pf_err = float(row[pf_err_col]) * scale if pf_err_col in row and not pd.isna(row[pf_err_col]) else np.nan

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
        precision: int = 2,
        math_delimiters: bool = True,
    ) -> str:
        if pd.isna(val):
            return "-"

        if pd.isna(err):
            content = f"{val:.{precision}f}"
        else:
            content = f"{val:.{precision}f} \\pm {err:.{precision}f}"

        return f"${content}$" if math_delimiters else content

    def build_et_grid_table(
        self,
        strategy: str = "strategy1",
        as_percentage: bool = True,
        math_delimiters: bool = True,
        precision: int = 2,
        use_latex_labels: bool = True,
        include_global: bool = True,
    ) -> pd.DataFrame:
        """Constructs a 2D grid table with ET interval column blocks and SP, PF, PD sub-columns.

        Args:
            strategy: Target strategy ('strategy1' or 'strategy2').
            as_percentage: Multiply values and uncertainties by 100.
            math_delimiters: Enclose cell strings with dollar signs.
            precision: Decimal precision (default: 2 for percentages).
            use_latex_labels: Use LaTeX formatted interval descriptions.
            include_global: Append composite GLOBAL row at the bottom.

        Returns:
            DataFrame with MultiIndex columns (ET_Interval, Metric) and eta intervals as rows.
        """
        numeric_df = self.build_numeric_table(
            strategy=strategy,
            as_percentage=as_percentage,
            include_global=include_global,
        )

        et_labels = self.ET_LABELS_LATEX if use_latex_labels else self.ET_LABELS_PLAIN
        eta_labels = self.ETA_LABELS_LATEX if use_latex_labels else self.ETA_LABELS_PLAIN

        available_ets = sorted([int(x) for x in numeric_df[numeric_df["iet"] >= 0]["iet"].unique()])
        available_etas = sorted([int(x) for x in numeric_df[numeric_df["ieta"] >= 0]["ieta"].unique()])

        metric_order = ["SP", "PF", "PD"]

        column_tuples: List[Tuple[str, str]] = []
        for iet in available_ets:
            et_label = et_labels.get(iet, f"ET {iet}")
            for metric in metric_order:
                column_tuples.append((et_label, metric))

        multi_cols = pd.MultiIndex.from_tuples(column_tuples, names=["ET Interval", "Metric"])

        row_indices: List[str] = [eta_labels.get(ieta, f"eta {ieta}") for ieta in available_etas]
        has_global = (numeric_df["iet"] < 0).any()
        if include_global and has_global:
            row_indices.append("GLOBAL" if not use_latex_labels else "$\\text{GLOBAL}$")

        grid_data: List[List[str]] = []
        for ieta in available_etas:
            row_data: List[str] = []
            for iet in available_ets:
                sub = numeric_df[(numeric_df["iet"] == iet) & (numeric_df["ieta"] == ieta)]
                if len(sub) > 0:
                    r = sub.iloc[0]
                    sp_str = self._format_cell(r["sp"], r["sp_uncertainty"], precision, math_delimiters)
                    pf_str = self._format_cell(r["pf"], r["pf_uncertainty"], precision, math_delimiters)
                    pd_str = self._format_cell(r["pd"], r["pd_uncertainty"], precision, math_delimiters)
                else:
                    sp_str, pf_str, pd_str = "-", "-", "-"
                row_data.extend([sp_str, pf_str, pd_str])
            grid_data.append(row_data)

        if include_global and has_global:
            global_sub = numeric_df[numeric_df["iet"] < 0]
            r = global_sub.iloc[0]
            sp_str = self._format_cell(r["sp"], r["sp_uncertainty"], precision, math_delimiters)
            pf_str = self._format_cell(r["pf"], r["pf_uncertainty"], precision, math_delimiters)
            pd_str = self._format_cell(r["pd"], r["pd_uncertainty"], precision, math_delimiters)
            row_data = []
            for _ in available_ets:
                row_data.extend([sp_str, pf_str, pd_str])
            grid_data.append(row_data)

        index_name = "$|\\eta|\\text{ Range}$" if use_latex_labels else "eta Range"
        df_grid = pd.DataFrame(grid_data, index=pd.Index(row_indices, name=index_name), columns=multi_cols)
        return df_grid

    def build_latex_formatted_table(
        self,
        strategy: str = "strategy1",
        as_percentage: bool = True,
        math_delimiters: bool = True,
        precision: int = 2,
        include_global: bool = True,
    ) -> pd.DataFrame:
        """Constructs 25-region summary table with LaTeX-formatted string cells.

        Args:
            strategy: Target strategy ('strategy1' or 'strategy2').
            as_percentage: Multiply values and uncertainties by 100.
            math_delimiters: Enclose cell strings with dollar signs.
            precision: Decimal precision for numerical figures.
            include_global: Whether to append the global composite row.

        Returns:
            DataFrame with formatted strings for SP, PF, and PD.
        """
        numeric_df = self.build_numeric_table(
            strategy=strategy,
            as_percentage=as_percentage,
            include_global=include_global,
        )

        formatted_records = []
        for _, row in numeric_df.iterrows():
            formatted_records.append(
                {
                    "Region": row["region"],
                    "iet": row["iet"],
                    "ieta": row["ieta"],
                    "SP": self._format_cell(row["sp"], row["sp_uncertainty"], precision, math_delimiters),
                    "PF": self._format_cell(row["pf"], row["pf_uncertainty"], precision, math_delimiters),
                    "PD": self._format_cell(row["pd"], row["pd_uncertainty"], precision, math_delimiters),
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

        if isinstance(df.columns, pd.MultiIndex):
            flat_cols = [f"{col[0]} - {col[1]}" for col in df.columns]
            export_df = df.copy()
            export_df.columns = flat_cols
            export_df.to_csv(path, index=True)
        else:
            df.to_csv(path, index=False)
        return path

    def to_latex_snippet(
        self,
        df: pd.DataFrame,
        caption: Optional[str] = None,
        label: Optional[str] = None,
        resize_box: bool = True,
        interval_pipe: bool = True,
    ) -> str:
        """Converts formatted DataFrame to a LaTeX tabular snippet.

        Args:
            df: Formatted DataFrame (single index or MultiIndex).
            caption: Optional table caption.
            label: Optional table reference label.
            resize_box: Whether to wrap wide tables with resizebox.
            interval_pipe: Whether to place vertical pipes between energy intervals.

        Returns:
            String containing the formatted LaTeX code block.
        """
        display_df = df.copy()

        if isinstance(display_df.columns, pd.MultiIndex):
            num_cols = len(display_df.columns)
            num_intervals = num_cols // 3
            if interval_pipe:
                col_format = "l | " + " | ".join(["ccc"] * num_intervals)
                multicolumn_format = "c|"
            else:
                col_format = "l " + " ".join(["ccc"] * num_intervals)
                multicolumn_format = "c"

            latex_body = display_df.to_latex(
                index=True,
                escape=False,
                multicolumn=True,
                multicolumn_format=multicolumn_format,
                column_format=col_format,
            )

            if interval_pipe:
                lines = latex_body.splitlines()
                for idx, line in enumerate(lines):
                    if "\\multicolumn" in line:
                        r_idx = line.rfind("{c|}")
                        if r_idx != -1:
                            lines[idx] = line[:r_idx] + "{c}" + line[r_idx + 4:]
                        break
                latex_body = "\n".join(lines) + "\n"
        else:
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

        if resize_box and isinstance(df.columns, pd.MultiIndex):
            wrapped_body = (
                "\\resizebox{\\textwidth}{!}{%\n"
                f"{latex_body}"
                "}\n"
            )
        else:
            wrapped_body = latex_body

        return (
            "\\begin{table}[htbp]\n"
            "\\centering\n"
            f"{caption_block}"
            f"{label_block}"
            f"{wrapped_body}"
            "\\end{table}\n"
        )
