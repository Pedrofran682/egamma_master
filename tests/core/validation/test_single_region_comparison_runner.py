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
    assert (tables_dir / "et_1_eta_1_best_sp_vs_hard_cut_strategy_metrics.csv").exists()
    assert len(saved_quad["histograms"]) == 2

    sig_js_df, fig_sig = runner.run_variable_analysis(payload, out_dir, tables_dir, is_signal=True)
    assert not sig_js_df.empty
    assert (out_dir / "Variables_Signal" / "trig_L2_calo_weta2_signal_selection.png").exists()
    assert fig_sig is not None
