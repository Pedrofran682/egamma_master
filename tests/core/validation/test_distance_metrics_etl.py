import pathlib
import pandas as pd
import pytest

from src.core.Validation.DistanceMetricsETL import DistanceMetricsETL


@pytest.fixture
def temp_validation_dir(tmp_path: pathlib.Path) -> pathlib.Path:
    exec_dir = tmp_path / "SingleRegionValidation_config_nameModelV9_HighBatch_100Rings_id20261029001623"
    tables_dir = exec_dir / "Tables"
    tables_dir.mkdir(parents=True)

    dummy_df1 = pd.DataFrame({
        "Feature": ["trig_L2_calo_et", "trig_L2_calo_eta"],
        "KS Distance (D)": [0.05, 0.08],
        "Wasserstein Distance": [0.12, 0.15],
    })
    dummy_df2 = pd.DataFrame({
        "Feature": ["trig_L2_calo_et", "trig_L2_calo_eta"],
        "KS Distance (D)": [0.03, 0.04],
        "Wasserstein Distance": [0.09, 0.10],
    })

    file1 = tables_dir / "et_1_eta_1_distance_metrics_model1_vs_model2_background.csv"
    file2 = tables_dir / "et_2_eta_3_distance_metrics_vs_true_signal.csv"

    dummy_df1.to_csv(file1, index=False)
    dummy_df2.to_csv(file2, index=False)

    return exec_dir


def test_extract_et_eta():
    file_name = "et_1_eta_1_distance_metrics_model1_vs_model2_background.csv"
    et, eta = DistanceMetricsETL.extract_et_eta(file_name)
    assert et == 1
    assert eta == 1

    file_name_compact = "et2_eta3_distance_metrics_test.csv"
    et2, eta3 = DistanceMetricsETL.extract_et_eta(file_name_compact)
    assert et2 == 2
    assert eta3 == 3

    with pytest.raises(ValueError):
        DistanceMetricsETL.extract_et_eta("invalid_file_name.csv")


def test_extract_execution_folder(tmp_path: pathlib.Path):
    exec_folder_name = "SingleRegionValidation_TestExecution_id123456"
    file_path = tmp_path / exec_folder_name / "Tables" / "et_0_eta_0_distance_metrics_test.csv"
    file_path.parent.mkdir(parents=True)
    file_path.touch()

    folder = DistanceMetricsETL.extract_execution_folder(file_path)
    assert folder == exec_folder_name


def test_extract_metric_type():
    file_name = "et_1_eta_1_distance_metrics_model1_vs_model2_background.csv"
    metric_type = DistanceMetricsETL.extract_metric_type(file_name)
    assert metric_type == "model1_vs_model2_background"


def test_etl_pipeline(temp_validation_dir: pathlib.Path):
    etl = DistanceMetricsETL(folder_paths=[temp_validation_dir])
    assert len(etl.discovered_files) == 2

    consolidated_df = etl.extract_and_consolidate()
    assert len(consolidated_df) == 4
    assert "ET" in consolidated_df.columns
    assert "ETA" in consolidated_df.columns
    assert "ExecutionFolder" in consolidated_df.columns
    assert "MetricType" in consolidated_df.columns

    assert set(consolidated_df["ET"].unique()) == {1, 2}
    assert set(consolidated_df["ETA"].unique()) == {1, 3}
    assert consolidated_df["ExecutionFolder"].iloc[0] == temp_validation_dir.name

    summary = etl.get_summary(consolidated_df)
    assert not summary.empty
    assert len(summary) == 2


def test_evaluate_distribution_equivalence():
    test_df = pd.DataFrame({
        "KS Distance (D)": [0.01, 0.25],
        "KS Uncertainty": [0.008, 0.005],
        "p-value": [0.10, 1e-10],
        "Wasserstein Distance": [0.01, 0.20],
        "Wasserstein Uncertainty": [0.008, 0.005],
        "KL Divergence": [0.005, 0.15],
        "KL Uncertainty": [0.003, 0.008],
        "JS Divergence": [0.002, 0.08],
        "JS Uncertainty": [0.001, 0.004],
    })

    result = DistanceMetricsETL.evaluate_distribution_equivalence(test_df)

    assert "KS_Equivalent" in result.columns
    assert "Wasserstein_Equivalent" in result.columns
    assert "KL_Equivalent" in result.columns
    assert "JS_Equivalent" in result.columns

    assert result["KS_Equivalent"].tolist() == [True, False]
    assert result["Wasserstein_Equivalent"].tolist() == [True, False]
    assert result["KL_Equivalent"].tolist() == [True, False]
    assert result["JS_Equivalent"].tolist() == [True, False]


def test_format_latex_table():
    test_df = pd.DataFrame({
        "Feature": ["trig_L2_calo_et"],
        "Comparison": ["Model1 vs Model2"],
        "KS Distance (D)": [0.0123],
        "KS Uncertainty": [0.0045],
        "Wasserstein Distance": [0.0678],
        "Wasserstein Uncertainty": [0.0091],
        "KL Divergence": [0.0022],
        "KL Uncertainty": [0.0008],
        "JS Divergence": [0.0011],
        "JS Uncertainty": [0.0004],
    })

    formatted_df, latex_str = DistanceMetricsETL.format_latex_table(test_df, precision=4)

    assert "Feature" in formatted_df.columns
    assert "Comparison" in formatted_df.columns
    assert formatted_df["Feature"].iloc[0] == r"trig\_L2\_calo\_et"
    assert formatted_df["KS Distance ($D$)"].iloc[0] == r"$0.0123 \pm 0.0045$"
    assert r"\begin{tabular}" in latex_str
    assert r"$0.0123 \pm 0.0045$" in latex_str


def test_format_table_with_uncertainties():
    custom_df = pd.DataFrame({
        "Operating_Point": ["Loose", "Tight"],
        "Signal_Eff": [0.9424, 0.8812],
        "Signal_Eff_Unc": [0.0015, 0.0020],
        "Fake_Rate": [0.0531, 0.0210],
        "Fake_Rate_Unc": [0.0008, 0.0005],
    })

    # Test with 3-element tuples (display_name, val_col, unc_col)
    pairs_3 = [
        (r"$\varepsilon_{\text{sig}}$", "Signal_Eff", "Signal_Eff_Unc"),
        (r"Fake Rate", "Fake_Rate", "Fake_Rate_Unc"),
    ]
    fmt_df_3, latex_3 = DistanceMetricsETL.format_table_with_uncertainties(
        custom_df, pairs=pairs_3, id_cols=["Operating_Point"], precision=3
    )

    assert "Operating_Point" in fmt_df_3.columns
    assert fmt_df_3["Operating_Point"].iloc[0] == "Loose"
    assert fmt_df_3[r"$\varepsilon_{\text{sig}}$"].iloc[0] == r"$0.942 \pm 0.002$"
    assert r"$0.053 \pm 0.001$" in latex_3

    # Test with 2-element tuples (val_col, unc_col)
    pairs_2 = [
        ("Signal_Eff", "Signal_Eff_Unc"),
    ]
    fmt_df_2, latex_2 = DistanceMetricsETL.format_table_with_uncertainties(
        custom_df, pairs=pairs_2, id_cols=["Operating_Point"]
    )
    assert "Signal_Eff" in fmt_df_2.columns
    assert fmt_df_2["Signal_Eff"].iloc[0] == r"$0.9424 \pm 0.0015$"

