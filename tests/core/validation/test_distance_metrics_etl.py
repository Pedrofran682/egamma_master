import logging
import pathlib
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.core.Validation import LatexTableFormatter as ExportedLatexTableFormatter
from src.core.Validation.DistanceMetricsETL import DistanceMetricsETL
from src.core.Validation.HoldoutEvaluator import HoldoutEvaluationResult, HoldoutEvaluator
from src.core.Validation.LatexTableFormatter import LatexTableFormatter
from src.core.Validation.PileupAnalyzer import PileupAnalyzer
from src.core.Validation.QuadrantAnalyzer import QuadrantAnalyzer
from src.utils import (
    _compute_shared_bin_edges,
    compute_js_divergence_with_uncertainty,
    compute_kl_divergence_with_uncertainty,
    compute_sp_metric,
    get_best_sp_model,
)


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


def test_latex_table_formatter_exported_from_validation_package() -> None:
    assert ExportedLatexTableFormatter is LatexTableFormatter


def test_latex_table_formatter_value_with_uncertainty() -> None:
    assert (
        LatexTableFormatter.format_value_with_uncertainty(0.01234, 0.00456)
        == r"$0.0123 \pm 0.0046$"
    )
    assert (
        LatexTableFormatter.format_value_with_uncertainty(0.9424, 0.0035, precision=2, bold=True)
        == r"$\mathbf{0.94 \pm 0.00}$"
    )
    assert (
        LatexTableFormatter.format_value_with_uncertainty(
            94.24, 0.35, precision=2, decimal_comma=True, math_delimiters=False
        )
        == r"94{,}24 \pm 0{,}35"
    )
    assert LatexTableFormatter.format_value_with_uncertainty(float("nan"), 0.01) == "N/A"
    assert LatexTableFormatter.format_value_with_uncertainty(0.95, float("nan")) == "N/A"
    assert LatexTableFormatter.format_value_with_uncertainty(float("inf"), 0.01) == "N/A"
    assert (
        LatexTableFormatter.format_value_with_uncertainty(
            94.5, float("nan"), precision=2, allow_missing_err=True
        )
        == "$94.50$"
    )


def test_latex_table_formatter_wrap_booktabs_environment() -> None:
    tabular = "\\begin{tabular}{cc}\na & b \\\\\n\\end{tabular}"
    wrapped = LatexTableFormatter.wrap_booktabs_environment(
        tabular, caption="Test Caption", label="tab:test", resize_to_textwidth=True
    )
    assert "\\begin{table}[htbp]" in wrapped
    assert "\\centering" in wrapped
    assert "\\caption{Test Caption}" in wrapped
    assert "\\label{tab:test}" in wrapped
    assert "\\resizebox{\\textwidth}{!}{%" in wrapped
    assert "\\end{table}" in wrapped

    unresized = LatexTableFormatter.wrap_booktabs_environment(
        tabular, caption="Plain", label="tab:plain", resize_to_textwidth=False
    )
    assert "\\resizebox" not in unresized
    assert "\\caption{Plain}" in unresized


def test_latex_table_formatter_pdg_and_dataframe_methods() -> None:
    assert LatexTableFormatter.format_pm(0.008336, 0.005644, sig_figs=2) == r"$0.0083 \pm 0.0056$"
    assert LatexTableFormatter.format_pm(16.628878, 23.797069, sig_figs=2) == r"$17 \pm 24$"
    assert LatexTableFormatter.format_pm(0.010359, 0.012521, sig_figs=2) == r"$0.010 \pm 0.013$"
    assert LatexTableFormatter.format_pm(1.23, 0.0) == r"$1.2300 \pm 0.0000$"
    assert LatexTableFormatter.format_pm(np.nan, 0.1) == "-"

    df = pd.DataFrame({
        "Feature": ["trig_L2_calo_et"],
        "KS_Distance": [0.008336],
        "KS_Uncertainty": [0.005644],
    })
    concat_df = LatexTableFormatter.concatenate_columns(
        df, "KS_Distance", "KS_Uncertainty", new_col="KS", sig_figs=2, drop_original=True
    )
    assert list(concat_df.columns) == ["Feature", "KS"]
    assert concat_df["KS"].iloc[0] == r"$0.0083 \pm 0.0056$"

    merged_df = LatexTableFormatter.merge_uncertainty_pairs(
        df,
        pairs={r"KS ($D_{\text{KS}}$)": ("KS_Distance", "KS_Uncertainty")},
        precision=4,
        drop_original=True,
    )
    assert merged_df[r"KS ($D_{\text{KS}}$)"].iloc[0] == r"$0.0083 \pm 0.0056$"

    latex_output = LatexTableFormatter.to_latex(
        merged_df,
        caption="Distance Metrics",
        label="tab:dist",
        resize_box=False,
    )
    assert r"trig\_L2\_calo\_et" in latex_output
    assert r"$D_{\text{KS}}$" in latex_output
    assert r"\caption{Distance Metrics}" in latex_output
    assert r"\label{tab:dist}" in latex_output


def test_compute_sp_metric_scalar_and_array() -> None:
    assert compute_sp_metric(0.9, 0.1) == pytest.approx(0.9)
    assert compute_sp_metric(1.0, 0.0) == pytest.approx(1.0)
    assert compute_sp_metric(0.0, 1.0) == pytest.approx(0.0)
    assert compute_sp_metric(-0.1, 0.2) == 0.0
    assert compute_sp_metric(0.9, 1.2) == 0.0

    pd_arr = np.array([0.9, 1.0, 0.0, -0.1])
    pf_arr = np.array([0.1, 0.0, 1.0, 0.2])
    sp_arr = compute_sp_metric(pd_arr, pf_arr)
    assert isinstance(sp_arr, np.ndarray)
    np.testing.assert_allclose(sp_arr, np.array([0.9, 1.0, 0.0, 0.0]))


def test_compute_shared_bin_edges_and_divergences() -> None:
    assert _compute_shared_bin_edges(np.array([]), np.array([1.0]), 10) is None
    assert _compute_shared_bin_edges(np.array([1.0]), np.array([]), 10) is None
    assert _compute_shared_bin_edges(np.array([1.0]), np.array([1.0]), 0) is None

    const_edges = _compute_shared_bin_edges(np.array([5.0, 5.0]), np.array([5.0]), 4)
    assert const_edges is not None
    assert len(const_edges) == 5
    assert const_edges[0] == pytest.approx(4.0)
    assert const_edges[-1] == pytest.approx(6.0)

    rng = np.random.default_rng(7)
    p = rng.normal(0.0, 1.0, size=200)
    q = rng.normal(0.5, 1.0, size=200)
    edges = _compute_shared_bin_edges(p, q, 20)
    assert edges is not None
    assert len(edges) == 21
    assert np.all(np.diff(edges) > 0)

    kl_val, kl_unc = compute_kl_divergence_with_uncertainty(p, q, bins=20, n_bootstraps=10)
    js_val, js_unc = compute_js_divergence_with_uncertainty(p, q, bins=20, n_bootstraps=10)
    assert kl_val > 0.0
    assert kl_unc >= 0.0
    assert 0.0 < js_val <= 1.0
    assert js_unc >= 0.0
    assert compute_kl_divergence_with_uncertainty(np.array([]), q) == (0.0, 0.0)
    assert compute_js_divergence_with_uncertainty(p, np.array([])) == (0.0, 0.0)


def test_get_best_sp_model_empty_dataframe_logs_english(caplog: pytest.LogCaptureFixture) -> None:
    empty_df = pd.DataFrame()
    with caplog.at_level(logging.ERROR):
        model, info = get_best_sp_model(empty_df, "ModelV1", 100)
    assert model is None
    assert info is None
    assert "Training results list is empty." in caplog.text


def test_pileup_analyzer_extract_feature_delegation() -> None:
    feature_names = np.array(["avgmu", "trig_L2_calo_emaxs1", "trig_L2_calo_e2tsts1"])
    features = np.array([
        [25.0, 120.0, 30.0],
        [40.0, 0.0, 0.0],
    ])
    np.testing.assert_allclose(PileupAnalyzer.extract_feature(features, feature_names, "AVGMU"), [25.0, 40.0])
    np.testing.assert_allclose(
        PileupAnalyzer.extract_feature(features, feature_names, "Eratio"),
        [(120.0 - 30.0) / 150.0, 0.0],
    )
    with pytest.raises(KeyError):
        PileupAnalyzer.extract_feature(features, feature_names, "unknown_col")


def test_quadrant_analyzer_get_target_pd_and_eratio_delegation(tmp_path: pathlib.Path) -> None:
    ref_df = pd.DataFrame({
        "et_bin": [1, 2],
        "eta_bin": [1, 1],
        "eff_sig_loose": [0.95, 0.96],
        "eff_sig_medium": [0.91, 0.92],
        "eff_sig_tight": [0.87, 0.88],
    })
    with patch.object(QuadrantAnalyzer, "_load_config", return_value=MagicMock()):
        with patch("src.core.Validation.QuadrantAnalyzer.NeuralRingerTrainer"):
            with patch("src.core.Validation.QuadrantAnalyzer.SplitManifest"):
                with patch("src.core.Validation.QuadrantAnalyzer.ResultAggregator"):
                    analyzer_tight = QuadrantAnalyzer(
                        config_path1="dummy.yaml",
                        data_path1=tmp_path,
                        mode="model_vs_cut",
                        working_point="tight",
                        default_target_pd=0.9424,
                    )
                    analyzer_tight.reference_eff_df = ref_df
                    assert analyzer_tight._get_target_pd(1, 1) == pytest.approx(0.87)
                    assert analyzer_tight._get_target_pd(9, 9) == pytest.approx(0.9424)

                    analyzer_tight.working_point = "medium"
                    assert analyzer_tight._get_target_pd(2, 1) == pytest.approx(0.92)

                    analyzer_tight.working_point = "nonexistent_wp"
                    assert analyzer_tight._get_target_pd(1, 1) == pytest.approx(0.95)

    shapes_missing = QuadrantAnalyzer.extract_l2_showershapes(
        np.array([[0.02]]), ["trig_L2_calo_weta2"]
    )
    assert "trig_L2_calo_weta2" in shapes_missing
    assert "Eratio" not in shapes_missing


def test_holdout_evaluator_probs_to_logits_and_sp() -> None:
    probs = np.array([0.0, 0.5, 1.0])
    logits = HoldoutEvaluator._probs_to_logits(probs)
    assert logits[1] == pytest.approx(0.0)
    assert logits[0] == pytest.approx(np.log(1e-7 / (1.0 - 1e-7)))
    assert logits[2] == pytest.approx(np.log((1.0 - 1e-7) / 1e-7))

    result = HoldoutEvaluationResult(
        holdout_features_full=np.zeros((4, 2)),
        holdout_labels=np.array([1, 1, 0, 0]),
        predictions_default_threshold=np.array([1, 1, 0, 0]),
        predictions_calibrated_cut=np.array([1, 1, 0, 0]),
        predicted_probabilities=np.array([0.8, 0.9, 0.1, 0.2]),
        accuracy_default_threshold=1.0,
        signal_efficiency_default_threshold=1.0,
        background_rejection_default_threshold=1.0,
        false_positive_rate=0.0,
        false_negative_rate=0.0,
        signal_cutoff_percentile=5.0,
        calibrated_decision_threshold=0.5,
        false_alarm_probability=0.0,
    )
    np.testing.assert_allclose(
        result.logits, HoldoutEvaluator._probs_to_logits(result.predicted_probabilities)
    )
    assert result.sp == pytest.approx(1.0)


