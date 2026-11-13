from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.core.Validation.RegionalOperatingPointsConsolidator import (
    RegionalOperatingPointsConsolidator,
)


@pytest.fixture
def mock_consolidator(tmp_path: Path) -> RegionalOperatingPointsConsolidator:
    class DummyConsolidator(RegionalOperatingPointsConsolidator):
        def __init__(self) -> None:
            self.target_pd = 0.9424
            self.working_point = "loose"
            self.method_aliases = {
                "calibrated_hard_cut": "Calibrated Hard Cut (Target PD: 94.24%)",
                "pileup_corrected_cut": "Pileup-Corrected Cut T(mu) (Target PD)",
                "athena_cut": "Athena Loose Cut",
            }
            self._records = []
            for et in range(2):
                for eta in range(2):
                    for m_key in self.DEFAULT_METHOD_KEYS:
                        self._records.append(
                            {
                                "iet": et,
                                "ieta": eta,
                                "method_key": m_key,
                                "method": self.method_aliases[m_key],
                                "target_pd": 0.9424,
                                "decision_threshold": "0.1000",
                                "pd": 0.9424,
                                "pd_uncertainty": 0.0035,
                                "sp": 0.9500 if m_key == "pileup_corrected_cut" else 0.9400,
                                "sp_uncertainty": 0.0030,
                                "pf": 0.0500,
                                "pf_uncertainty": 0.0020,
                                "eff": 0.9400,
                                "eff_uncertainty": 0.0025,
                                "bg_eff": 0.9500,
                                "bg_eff_uncertainty": 0.0020,
                                "tp": 942,
                                "fp": 50,
                                "tn": 950,
                                "fn": 58,
                                "total": 2000,
                            }
                        )

    return DummyConsolidator()


def test_build_tidy_dataframe(mock_consolidator: RegionalOperatingPointsConsolidator) -> None:
    df = mock_consolidator.build_tidy_dataframe()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2 * 2 * 3
    assert "iet" in df.columns
    assert "ieta" in df.columns
    assert "method_key" in df.columns
    assert "sp" in df.columns


def test_build_grid_table(mock_consolidator: RegionalOperatingPointsConsolidator) -> None:
    grid = mock_consolidator.build_grid_table(as_percentage=True, precision=2)
    assert isinstance(grid.columns, pd.MultiIndex)
    assert isinstance(grid.index, pd.MultiIndex)
    assert list(grid.index.names) == ["eta", "Method"]
    assert len(grid.index) == 2 * 3
    assert len(grid.columns) == 2 * 3


def test_to_latex_snippet(mock_consolidator: RegionalOperatingPointsConsolidator) -> None:
    latex = mock_consolidator.to_latex_snippet(as_percentage=True, precision=2)
    assert "\\begin{table}" in latex
    assert "\\resizebox{\\textwidth}{!}" in latex
    assert "\\multirow{3}{*}{$\\eta_0$}" in latex
    assert "Method" in latex
    assert "Método" not in latex
    assert "$P_D[\\%]$" in latex
    assert "\\end{tabular}" in latex
    assert "\\end{table}" in latex


def test_export_parquet_and_latex(
    mock_consolidator: RegionalOperatingPointsConsolidator, tmp_path: Path
) -> None:
    parquet_path = tmp_path / "table.parquet"
    latex_path = tmp_path / "table.tex"

    saved_parquet = mock_consolidator.export_parquet(parquet_path)
    assert saved_parquet.exists()

    saved_latex = mock_consolidator.export_latex(latex_path)
    assert saved_latex.exists()
    content = saved_latex.read_text(encoding="utf-8")
    assert "\\begin{table}" in content
