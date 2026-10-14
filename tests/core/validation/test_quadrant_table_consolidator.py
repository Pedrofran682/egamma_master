import pathlib
import numpy as np
import pandas as pd
import pytest

from src.core.Validation.QuadrantTableConsolidator import QuadrantTableConsolidator


@pytest.fixture
def sample_quadrant_df() -> pd.DataFrame:
    rows = []
    for et in range(5):
        for eta in range(5):
            rows.append(
                {
                    "region": f"iet{et}.ieta{eta}",
                    "iet": et,
                    "ieta": eta,
                    "strategy1_name": "ModelV6",
                    "strategy2_name": "AthenaLoose",
                    "s1_pd": 0.9400 + et * 0.001,
                    "s1_sig_eff_uncertainty": 0.0020,
                    "s1_pf": 0.0500 + eta * 0.001,
                    "s1_bg_eff_uncertainty": 0.0015,
                    "s1_sp": 0.9450,
                    "s2_pd": 0.9300,
                    "s2_sig_eff_uncertainty": 0.0025,
                    "s2_pf": 0.0600,
                    "s2_bg_eff_uncertainty": 0.0020,
                    "s2_sp": 0.9350,
                }
            )
    rows.append(
        {
            "region": "GLOBAL",
            "iet": -1,
            "ieta": -1,
            "strategy1_name": "ModelV6",
            "strategy2_name": "AthenaLoose",
            "s1_pd": 0.9420,
            "s1_sig_eff_uncertainty": 0.0005,
            "s1_pf": 0.0520,
            "s1_bg_eff_uncertainty": 0.0004,
            "s1_sp": 0.9460,
            "s2_pd": 0.9320,
            "s2_sig_eff_uncertainty": 0.0006,
            "s2_pf": 0.0610,
            "s2_bg_eff_uncertainty": 0.0005,
            "s2_sp": 0.9360,
        }
    )
    return pd.DataFrame(rows)


def test_strategy_name_extraction(sample_quadrant_df: pd.DataFrame) -> None:
    consolidator = QuadrantTableConsolidator(sample_quadrant_df)
    assert consolidator.get_strategy_name("strategy1") == "ModelV6"
    assert consolidator.get_strategy_name("strategy2") == "AthenaLoose"


def test_build_numeric_table_no_sp_propagation(sample_quadrant_df: pd.DataFrame) -> None:
    consolidator = QuadrantTableConsolidator(sample_quadrant_df)
    df = consolidator.build_numeric_table(strategy="strategy1", include_global=True)

    assert len(df) == 26
    assert np.isnan(df.loc[0, "sp_uncertainty"])
    assert df.loc[0, "pd"] == pytest.approx(0.9400)
    assert df.loc[0, "pd_uncertainty"] == pytest.approx(0.0020)
    assert df.loc[0, "pf"] == pytest.approx(0.0500)
    assert df.loc[0, "pf_uncertainty"] == pytest.approx(0.0015)


def test_build_latex_formatted_table_with_math_delimiters(sample_quadrant_df: pd.DataFrame) -> None:
    consolidator = QuadrantTableConsolidator(sample_quadrant_df)
    df = consolidator.build_latex_formatted_table(
        strategy="strategy1",
        math_delimiters=True,
        precision=4,
        include_global=False,
    )

    assert len(df) == 25
    assert df.loc[0, "SP"] == "$0.9450$"
    assert df.loc[0, "PD"] == "$0.9400 \\pm 0.0020$"
    assert df.loc[0, "PF"] == "$0.0500 \\pm 0.0015$"


def test_build_latex_formatted_table_without_math_delimiters(sample_quadrant_df: pd.DataFrame) -> None:
    consolidator = QuadrantTableConsolidator(sample_quadrant_df)
    df = consolidator.build_latex_formatted_table(
        strategy="strategy2",
        math_delimiters=False,
        precision=3,
        include_global=True,
    )

    assert len(df) == 26
    assert df.loc[0, "SP"] == "0.935"
    assert df.loc[0, "PD"] == "0.930 \\pm 0.003"
    assert df.loc[0, "PF"] == "0.060 \\pm 0.002"


def test_export_csv_and_to_latex(tmp_path: pathlib.Path, sample_quadrant_df: pd.DataFrame) -> None:
    consolidator = QuadrantTableConsolidator(sample_quadrant_df)
    df = consolidator.build_latex_formatted_table(strategy="strategy1")

    output_csv = tmp_path / "table.csv"
    saved_path = consolidator.export_csv(df, output_csv)
    assert saved_path.exists()

    loaded_df = pd.read_csv(saved_path)
    assert len(loaded_df) == 26
    assert "SP" in loaded_df.columns
    assert "PD" in loaded_df.columns
    assert "PF" in loaded_df.columns

    latex_code = consolidator.to_latex_snippet(df, caption="Test Caption", label="tab:test")
    assert "\\begin{table}" in latex_code
    assert "\\caption{Test Caption}" in latex_code
    assert "\\label{tab:test}" in latex_code
    assert "\\end{table}" in latex_code
