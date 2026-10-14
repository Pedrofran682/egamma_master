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


def test_build_numeric_table_percentage(sample_quadrant_df: pd.DataFrame) -> None:
    consolidator = QuadrantTableConsolidator(sample_quadrant_df)
    df = consolidator.build_numeric_table(strategy="strategy1", as_percentage=True, include_global=True)

    assert len(df) == 26
    assert np.isnan(df.loc[0, "sp_uncertainty"])
    assert df.loc[0, "sp"] == pytest.approx(94.50)
    assert df.loc[0, "pd"] == pytest.approx(94.00)
    assert df.loc[0, "pd_uncertainty"] == pytest.approx(0.20)
    assert df.loc[0, "pf"] == pytest.approx(5.00)
    assert df.loc[0, "pf_uncertainty"] == pytest.approx(0.15)


def test_build_et_grid_table_shape_and_columns(sample_quadrant_df: pd.DataFrame) -> None:
    consolidator = QuadrantTableConsolidator(sample_quadrant_df)
    grid_df = consolidator.build_et_grid_table(
        strategy="strategy1",
        as_percentage=True,
        math_delimiters=True,
        precision=2,
        use_latex_labels=True,
        include_global=False,
    )

    assert grid_df.shape == (5, 15)
    first_col = grid_df.columns[0]
    assert first_col[1] == "SP"
    assert grid_df.columns[1][1] == "PF"
    assert grid_df.columns[2][1] == "PD"

    cell_sp = grid_df.iloc[0, 0]
    cell_pf = grid_df.iloc[0, 1]
    cell_pd = grid_df.iloc[0, 2]

    assert cell_sp == "$94.50$"
    assert cell_pf == "$5.00 \\pm 0.15$"
    assert cell_pd == "$94.00 \\pm 0.20$"


def test_build_et_grid_table_with_global(sample_quadrant_df: pd.DataFrame) -> None:
    consolidator = QuadrantTableConsolidator(sample_quadrant_df)
    grid_df = consolidator.build_et_grid_table(
        strategy="strategy1",
        as_percentage=True,
        include_global=True,
    )
    assert grid_df.shape == (6, 15)
    assert grid_df.index[-1] == "$\\text{GLOBAL}$"


def test_export_grid_csv_and_to_latex(tmp_path: pathlib.Path, sample_quadrant_df: pd.DataFrame) -> None:
    consolidator = QuadrantTableConsolidator(sample_quadrant_df)
    grid_df = consolidator.build_et_grid_table(strategy="strategy1", as_percentage=True)

    output_csv = tmp_path / "grid_table.csv"
    saved_path = consolidator.export_csv(grid_df, output_csv)
    assert saved_path.exists()

    loaded_df = pd.read_csv(saved_path)
    assert len(loaded_df) == 6
    assert any("SP" in c for c in loaded_df.columns)
    assert any("PF" in c for c in loaded_df.columns)
    assert any("PD" in c for c in loaded_df.columns)

    latex_code = consolidator.to_latex_snippet(
        grid_df,
        caption="Regional efficiencies across ET and eta",
        label="tab:regional_efficiencies",
        resize_box=True,
        interval_pipe=True,
    )
    assert "\\begin{table}" in latex_code
    assert "\\resizebox{\\textwidth}{!}" in latex_code
    assert "\\begin{tabular}{l | ccc | ccc | ccc | ccc | ccc}" in latex_code
    assert "\\multicolumn{3}{c|}" in latex_code
    assert "\\caption{Regional efficiencies across ET and eta}" in latex_code
    assert "\\label{tab:regional_efficiencies}" in latex_code
    assert "\\end{table}" in latex_code
