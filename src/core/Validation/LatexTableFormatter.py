import math
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


class LatexTableFormatter:
    """Formats DataFrames, metric uncertainty pairs, and tabular bodies into LaTeX tables."""

    @staticmethod
    def format_value_with_uncertainty(
        val: float,
        err: float,
        precision: int = 4,
        bold: bool = False,
        decimal_comma: bool = False,
        math_delimiters: bool = True,
        nan_placeholder: str = "N/A",
        allow_missing_err: bool = False,
    ) -> str:
        """Format a central value and uncertainty pair into a LaTeX math string.

        Args:
            val: Central numerical value.
            err: Associated uncertainty value.
            precision: Number of decimal places.
            bold: Whether to wrap the formatted content in bold math mode.
            decimal_comma: Whether to replace decimal dots with LaTeX decimal commas.
            math_delimiters: Whether to enclose the output in dollar math delimiters.
            nan_placeholder: Fallback string when required values are NaN or non-finite.
            allow_missing_err: Format only the central value when uncertainty is missing.

        Returns:
            Formatted LaTeX string or placeholder for missing values.
        """
        if pd.isna(val) or not np.isfinite(val):
            return nan_placeholder

        err_missing = pd.isna(err) or not np.isfinite(err)
        if err_missing and not allow_missing_err:
            return nan_placeholder

        val_str = f"{float(val):.{precision}f}"
        if decimal_comma:
            val_str = val_str.replace(".", "{,}")

        if err_missing:
            core = val_str
        else:
            err_str = f"{float(err):.{precision}f}"
            if decimal_comma:
                err_str = err_str.replace(".", "{,}")
            core = f"{val_str} \\pm {err_str}"

        if bold:
            core = f"\\mathbf{{{core}}}"

        return f"${core}$" if math_delimiters else core

    @staticmethod
    def wrap_booktabs_environment(
        tabular_body: str,
        caption: Optional[str] = None,
        label: Optional[str] = None,
        resize_to_textwidth: bool = True,
    ) -> str:
        """Wrap a LaTeX tabular block inside a centered table float environment.

        Args:
            tabular_body: Inner LaTeX tabular environment string.
            caption: Optional table caption.
            label: Optional table reference label.
            resize_to_textwidth: Whether to wrap the tabular block in resizebox.

        Returns:
            Complete LaTeX table float environment string.
        """
        body = tabular_body.strip()
        if resize_to_textwidth:
            body = f"\\resizebox{{\\textwidth}}{{!}}{{%\n{body}\n}}"

        lines: List[str] = ["\\begin{table}[htbp]", "\\centering"]
        if caption:
            lines.append(f"\\caption{{{caption}}}")
        if label:
            lines.append(f"\\label{{{label}}}")
        lines.append(body)
        lines.append("\\end{table}")
        return "\n".join(lines) + "\n"

    @staticmethod
    def escape_math_aware(text: Any) -> Any:
        """Escape underscores in plain text segments while preserving math-mode spans.

        Args:
            text: Input cell or header object.

        Returns:
            String with non-math underscores escaped, or the original object if non-string.
        """
        if not isinstance(text, str):
            return text
        parts = text.split("$")
        for i in range(0, len(parts), 2):
            parts[i] = parts[i].replace("_", r"\_")
        return "$".join(parts)

    @staticmethod
    def format_pm(
        val: float,
        unc: float,
        precision: Optional[int] = None,
        sig_figs: int = 2,
    ) -> str:
        """Format value and uncertainty as '$val \\pm unc$' with fixed or PDG rounding.

        Args:
            val: Central observable value.
            unc: Associated uncertainty.
            precision: Fixed decimal places, or None for PDG significant-figure rounding.
            sig_figs: Significant figures retained for uncertainty when precision is None.

        Returns:
            Formatted LaTeX math string '$val \\pm unc$', or '-' when non-finite.
        """
        if pd.isna(val) or pd.isna(unc) or not np.isfinite(val) or not np.isfinite(unc):
            return "-"

        if precision is not None:
            return LatexTableFormatter.format_value_with_uncertainty(
                val, unc, precision=precision, nan_placeholder="-"
            )

        if unc <= 0:
            return LatexTableFormatter.format_value_with_uncertainty(
                val, unc, precision=4, nan_placeholder="-"
            )

        order = math.floor(math.log10(abs(unc)))
        decimals = max(0, sig_figs - 1 - order)

        if round(abs(unc), decimals) >= 10 ** (order + 1):
            decimals = max(0, decimals - 1)

        if decimals == 0:
            return f"${round(val):.0f} \\pm {round(unc):.0f}$"

        return LatexTableFormatter.format_value_with_uncertainty(
            val, unc, precision=decimals, nan_placeholder="-"
        )

    @classmethod
    def concatenate_columns(
        cls,
        df: pd.DataFrame,
        val_col: str,
        unc_col: str,
        new_col: Optional[str] = None,
        precision: Optional[int] = None,
        sig_figs: int = 2,
        drop_original: bool = False,
    ) -> pd.DataFrame:
        """Concatenate a value column and uncertainty column into a '$val \\pm unc$' column.

        Args:
            df: Source DataFrame.
            val_col: Name of central value column.
            unc_col: Name of uncertainty column.
            new_col: Name of resulting merged column. Defaults to val_col.
            precision: Fixed decimal places (optional).
            sig_figs: Significant digits for uncertainty rounding.
            drop_original: Whether to drop val_col and unc_col from returned DataFrame.

        Returns:
            Enriched DataFrame copy with combined LaTeX column.
        """
        target_col = new_col if new_col is not None else val_col
        res = df.copy()

        res[target_col] = [
            cls.format_pm(v, u, precision=precision, sig_figs=sig_figs)
            for v, u in zip(res[val_col], res[unc_col])
        ]

        if drop_original:
            cols_to_drop = [c for c in [val_col, unc_col] if c != target_col]
            res = res.drop(columns=cols_to_drop)

        return res

    @classmethod
    def merge_uncertainty_pairs(
        cls,
        df: pd.DataFrame,
        pairs: Dict[str, Tuple[str, str]],
        precision: Optional[int] = None,
        sig_figs: int = 2,
        drop_original: bool = True,
    ) -> pd.DataFrame:
        """Merge multiple value/uncertainty column pairs in batch.

        Args:
            df: Source DataFrame.
            pairs: Mapping from output column name to (val_column, unc_column).
            precision: Optional fixed decimal precision.
            sig_figs: Significant figures for uncertainty rounding.
            drop_original: Whether to remove original unmerged columns.

        Returns:
            DataFrame copy with merged LaTeX columns.
        """
        res = df.copy()
        original_cols_to_drop = set()

        for new_col_name, (val_c, unc_c) in pairs.items():
            if val_c in res.columns and unc_c in res.columns:
                res[new_col_name] = [
                    cls.format_pm(v, u, precision=precision, sig_figs=sig_figs)
                    for v, u in zip(res[val_c], res[unc_c])
                ]
                original_cols_to_drop.update([val_c, unc_c])

        if drop_original:
            cols_to_drop = [c for c in original_cols_to_drop if c not in pairs]
            res = res.drop(columns=cols_to_drop, errors="ignore")

        return res

    @classmethod
    def to_latex(
        cls,
        df: pd.DataFrame,
        caption: Optional[str] = None,
        label: Optional[str] = None,
        column_format: Optional[str] = None,
        escape_underscores: bool = True,
        resize_box: bool = False,
        index: bool = False,
    ) -> str:
        """Transform a DataFrame into a LaTeX table snippet.

        Args:
            df: DataFrame to convert.
            caption: Optional table caption.
            label: Optional table label identifier.
            column_format: LaTeX column format string.
            escape_underscores: Whether to escape non-math underscores in text cells and headers.
            resize_box: Whether to wrap tabular in resizebox.
            index: Whether to display DataFrame index.

        Returns:
            LaTeX code block string.
        """
        display_df = df.copy()

        if escape_underscores:
            display_df.columns = [cls.escape_math_aware(str(c)) for c in display_df.columns]
            for col in display_df.select_dtypes(include=["object", "string"]):
                display_df[col] = display_df[col].apply(cls.escape_math_aware)

        if column_format is None:
            col_aligns = [
                "l"
                if (
                    display_df[c].dtype == "object"
                    or pd.api.types.is_string_dtype(display_df[c])
                )
                else "c"
                for c in display_df.columns
            ]
            column_format = " ".join(col_aligns)

        tabular_code = display_df.to_latex(
            index=index,
            escape=False,
            column_format=column_format,
        )

        if caption or label:
            return cls.wrap_booktabs_environment(
                tabular_body=tabular_code,
                caption=caption,
                label=label,
                resize_to_textwidth=resize_box,
            )

        if resize_box:
            return f"\\resizebox{{\\textwidth}}{{!}}{{\n{tabular_code.strip()}\n}}\n"

        return tabular_code
