from pathlib import Path
import numpy as np
import pytest

from src.core.Validation.SingleRegionComparisonRunner import (
    ComparisonModelConfig,
    ComparisonPayload,
    SingleRegionComparisonRunner,
)


def test_runner_validation_and_target_pd(tmp_path: Path) -> None:
    m1 = ComparisonModelConfig(
        config_path=tmp_path / "m1.yaml",
        results_path=tmp_path / "res1",
        alias="Model 1",
    )
    with pytest.raises(ValueError, match="Invalid target_pd_mode"):
        SingleRegionComparisonRunner(iet=1, ieta=1, model1=m1, target_pd_mode="bad")

    with pytest.raises(ValueError, match="`model2` must be provided"):
        SingleRegionComparisonRunner(iet=1, ieta=1, model1=m1, compare_same_model=False)

    runner = SingleRegionComparisonRunner(
        iet=1,
        ieta=1,
        model1=m1,
        compare_same_model=True,
        target_pd_mode="custom",
        custom_target_pd=0.9,
        language="pt",
    )
    pd_val, pd_tag = runner._resolve_target_pd(
        np.zeros((4, 3)), np.array([1, 1, 0, 0]), np.array(["a", "b", "c"])
    )
    assert np.isclose(pd_val, 0.9)
    assert "90.00%" in pd_tag
    assert "best_sp_vs_best_sp" in SingleRegionComparisonRunner.VALID_MODES
    assert "pileup_vs_pileup" in SingleRegionComparisonRunner.VALID_MODES

    athena_cfg = ComparisonModelConfig.athena_cut(working_point="loose")
    assert athena_cfg.method == "athena_cut"
    assert athena_cfg.alias == "Athena Cut (Loose)"
    assert athena_cfg.config_path is None
    assert athena_cfg.results_path is None

    with pytest.raises(ValueError, match="Invalid method"):
        ComparisonModelConfig(config_path=tmp_path / "m1.yaml", results_path=tmp_path / "res1", method="unknown")

    with pytest.raises(ValueError, match="required unless method == 'athena_cut'"):
        ComparisonModelConfig(alias="Missing paths", method="best_sp")


def test_runner_quadrant_and_variable_analysis(tmp_path: Path) -> None:
    rng = np.random.default_rng(7)
    n = 80
    labels = np.array([1] * 40 + [0] * 40)
    feats = np.column_stack([
        rng.normal(25.0, 3.0, n),
        rng.uniform(-1.0, 1.0, n),
        rng.normal(0.95, 0.02, n),
    ])
    feat_names = np.array(["trig_L2_calo_et", "trig_L2_calo_eta", "trig_L2_calo_weta2"])

    payload = ComparisonPayload(
        mode="best_sp_vs_hard_cut",
        region_key="et_1_eta_1",
        iet=1,
        ieta=1,
        strategy1_name="M1 [Best SP]",
        strategy2_name="M1 [Hard Cut]",
        preds_s1=np.array([1] * 35 + [0] * 5 + [1] * 5 + [0] * 35),
        preds_s2=np.array([1] * 32 + [0] * 8 + [1] * 3 + [0] * 37),
        thresh_s1=0.5,
        thresh_s2=0.6,
        probs_s1=rng.uniform(0, 1, n),
        probs_s2=rng.uniform(0, 1, n),
        test_features=feats,
        test_labels=labels,
        feature_names=feat_names,
        valid_features=["trig_L2_calo_et", "trig_L2_calo_weta2"],
        showershapes={"trig_L2_calo_weta2": feats[:, 2]},
        et_vals=feats[:, 0],
        eta_vals=feats[:, 1],
    )

    m1 = ComparisonModelConfig(config_path=tmp_path / "m1.yaml", results_path=tmp_path / "res1", alias="M1")
    runner = SingleRegionComparisonRunner(iet=1, ieta=1, model1=m1, language="pt", plot_format="png")

    out_dir = tmp_path / "out"
    tables_dir = tmp_path / "tables"
    strategy_df, counts_df, saved_quad = runner.run_quadrant_analysis(payload, out_dir, tables_dir)
    assert len(strategy_df) == 2
    assert len(counts_df) == 4
    assert (tables_dir / "et_1_eta_1_best_sp_vs_hard_cut_quadrant_strategy_metrics.csv").exists()
    assert (tables_dir / "et_1_eta_1_best_sp_vs_hard_cut_quadrant_counts.csv").exists()
    assert len(saved_quad["histograms"]) == 2
    assert all(
        "et_1_eta_1_best_sp_vs_hard_cut_quadrant_plot_" in Path(p).name
        for p in saved_quad["histograms"] + saved_quad["efficiency_curves"] + saved_quad["summary"]
    )

    sig_js_df, fig_sig = runner.run_variable_analysis(payload, out_dir, tables_dir, is_signal=True)
    assert not sig_js_df.empty
    assert (tables_dir / "et_1_eta_1_best_sp_vs_hard_cut_variable_js_divergence_signal.csv").exists()
    assert (
        out_dir / "Variables_Signal" / "et_1_eta_1_best_sp_vs_hard_cut_variable_plot_trig_L2_calo_weta2_signal_selection.png"
    ).exists()
    assert (
        out_dir / "Variables_Signal" / "et_1_eta_1_best_sp_vs_hard_cut_variable_plot_overview_grid_signal_selection.png"
    ).exists()
    assert fig_sig is not None


def test_runner_athena_reference_and_athena_model2(tmp_path: Path) -> None:
    rng = np.random.default_rng(11)
    n = 60
    labels = np.array([1] * 30 + [0] * 30)
    feats = np.column_stack([
        rng.normal(25.0, 3.0, n),
        rng.uniform(-1.0, 1.0, n),
        rng.normal(0.95, 0.02, n),
    ])
    feat_names = np.array(["trig_L2_calo_et", "trig_L2_calo_eta", "trig_L2_calo_weta2"])
    preds_athena = np.array([1] * 25 + [0] * 5 + [1] * 4 + [0] * 26)

    m1 = ComparisonModelConfig(config_path=tmp_path / "m1.yaml", results_path=tmp_path / "res1", alias="M1")
    runner = SingleRegionComparisonRunner(iet=1, ieta=1, model1=m1, language="pt", plot_format="png")

    payload_ml = ComparisonPayload(
        mode="best_sp_vs_hard_cut",
        region_key="et_1_eta_1",
        iet=1,
        ieta=1,
        strategy1_name="M1 [Best SP]",
        strategy2_name="M1 [Hard Cut]",
        preds_s1=np.array([1] * 27 + [0] * 3 + [1] * 2 + [0] * 28),
        preds_s2=np.array([1] * 26 + [0] * 4 + [1] * 3 + [0] * 27),
        thresh_s1=0.5,
        thresh_s2=0.6,
        probs_s1=rng.uniform(0, 1, n),
        probs_s2=rng.uniform(0, 1, n),
        test_features=feats,
        test_labels=labels,
        feature_names=feat_names,
        valid_features=["trig_L2_calo_weta2"],
        showershapes={"trig_L2_calo_weta2": feats[:, 2]},
        et_vals=feats[:, 0],
        eta_vals=feats[:, 1],
        preds_athena=preds_athena,
        athena_name="Athena Cut (Loose)",
        is_s1_ml=True,
        is_s2_ml=True,
    )
    js_with_ref, _ = runner.run_variable_analysis(payload_ml, tmp_path / "out_ref", tmp_path / "tab_ref", is_signal=True)
    assert "Athena Cut (Loose)" in set(js_with_ref["Strategy"])
    assert (
        tmp_path
        / "out_ref"
        / "Variables_Signal"
        / "et_1_eta_1_best_sp_vs_hard_cut_with_athena_cut_loose_ref_variable_plot_trig_L2_calo_weta2_signal_selection.png"
    ).exists()

    m2_ath = ComparisonModelConfig.athena_cut(working_point="loose")
    runner_ath = SingleRegionComparisonRunner(
        iet=1, ieta=1, model1=m1, model2=m2_ath, compare_same_model=False, working_point="loose"
    )
    assert runner_ath.build_comparison_tag("best_sp_vs_best_sp") == "best_sp_vs_athena_cut_loose"
    assert runner_ath.build_output_dir(tmp_path, "best_sp_vs_best_sp") == (
        tmp_path
        / "M1_vs_Athena_Cut_Loose"
        / "Comparison_best_sp_vs_athena_cut_loose_cross_et_1_eta_1"
    )

    payload_athena_s2 = ComparisonPayload(
        mode="best_sp_vs_hard_cut",
        region_key="et_1_eta_1",
        iet=1,
        ieta=1,
        strategy1_name="M1 [Best SP]",
        strategy2_name="Athena Cut (Loose)",
        preds_s1=np.array([1] * 27 + [0] * 3 + [1] * 2 + [0] * 28),
        preds_s2=preds_athena,
        thresh_s1=0.5,
        thresh_s2=0.0,
        probs_s1=rng.uniform(0, 1, n),
        probs_s2=None,
        test_features=feats,
        test_labels=labels,
        feature_names=feat_names,
        valid_features=["trig_L2_calo_weta2"],
        showershapes={"trig_L2_calo_weta2": feats[:, 2]},
        et_vals=feats[:, 0],
        eta_vals=feats[:, 1],
        preds_athena=preds_athena,
        athena_name="Athena Cut (Loose)",
        is_s1_ml=True,
        is_s2_ml=False,
    )
    strategy_df, counts_df, _ = runner.run_quadrant_analysis(
        payload_athena_s2, tmp_path / "out_ath", tmp_path / "tab_ath"
    )
    assert list(strategy_df["Strategy"]) == ["M1 [Best SP]", "Athena Cut (Loose)"]
    assert len(counts_df) == 4
    js_no_dup, _ = runner.run_variable_analysis(
        payload_athena_s2, tmp_path / "out_ath", tmp_path / "tab_ath", is_signal=True
    )
    assert list(js_no_dup["Strategy"]).count("Athena Cut (Loose)") == 1


def test_runner_pileup_analysis(tmp_path: Path) -> None:
    from src.core.Validation.PileupThresholdCalibrator import PileupThresholdCalibrator

    rng = np.random.default_rng(21)
    n = 120
    labels = np.array([1] * 60 + [0] * 60)
    mu = rng.uniform(10.0, 70.0, n)
    logits = rng.normal(1.0, 1.5, n)
    cal = PileupThresholdCalibrator(bins_mu=5, n_bootstraps=10, random_seed=42).calibrate_working_point(
        mu_values=mu[labels == 1],
        discriminant_values=logits[labels == 1],
        target_pd=0.9,
        working_point_name="Target",
    )

    m1 = ComparisonModelConfig(config_path=tmp_path / "m1.yaml", results_path=tmp_path / "res1", alias="M1")
    runner = SingleRegionComparisonRunner(iet=1, ieta=1, model1=m1, bins_mu=5, language="pt", plot_format="png")

    payload = ComparisonPayload(
        mode="best_sp_vs_pileup",
        region_key="et_1_eta_1",
        iet=1,
        ieta=1,
        strategy1_name="M1 [Best SP]",
        strategy2_name="M1 [T(mu)]",
        preds_s1=(logits >= 0.0).astype(int),
        preds_s2=(logits >= cal.slope * mu + cal.intercept).astype(int),
        thresh_s1=0.5,
        thresh_s2=float(cal.intercept),
        probs_s1=1.0 / (1.0 + np.exp(-logits)),
        probs_s2=1.0 / (1.0 + np.exp(-logits)),
        test_features=np.zeros((n, 1)),
        test_labels=labels,
        feature_names=np.array(["avgmu"]),
        valid_features=[],
        showershapes={},
        et_vals=np.ones(n) * 25.0,
        eta_vals=np.zeros(n),
        preds_athena=(logits >= 0.2).astype(int),
        athena_name="Athena Cut (Loose)",
        is_s1_ml=True,
        is_s2_ml=True,
        mu_test=mu,
        pileup_calibrations=[{
            "tag": "s2",
            "strategy_name": "M1 [T(mu)]",
            "calibration": cal,
            "mu_signal_calib": mu[labels == 1],
            "logits_signal_calib": logits[labels == 1],
        }],
    )

    out_dir = tmp_path / "out_pileup"
    tables_dir = tmp_path / "tab_pileup"
    fit_df, figs = runner.run_pileup_analysis(payload, out_dir, tables_dir)
    assert len(fit_df) == 1
    assert len(figs["calibration_figs"]) == 1
    assert len(figs["efficiency_figs"]) == 2
    assert (tables_dir / "et_1_eta_1_best_sp_vs_pileup_pileup_fit_parameters.csv").exists()
    assert (tables_dir / "et_1_eta_1_best_sp_vs_pileup_pileup_efficiency_vs_mu_signal.csv").exists()
    assert (tables_dir / "et_1_eta_1_best_sp_vs_pileup_pileup_efficiency_vs_mu_background.csv").exists()
    assert (out_dir / "Pileup" / "et_1_eta_1_best_sp_vs_pileup_pileup_2d_threshold_calibration_s2_plot.png").exists()
    assert (out_dir / "Pileup" / "et_1_eta_1_best_sp_vs_pileup_pileup_efficiency_vs_mu_signal_plot.png").exists()
    assert (out_dir / "Pileup" / "et_1_eta_1_best_sp_vs_pileup_pileup_efficiency_vs_mu_background_plot.png").exists()
