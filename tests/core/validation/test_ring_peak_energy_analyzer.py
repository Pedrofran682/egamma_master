import pathlib

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from src.core.Datasets.EgammaNpzDataset import EgammaNpzDataset
from src.core.Plotting.Context import RegionPlotContext
from src.core.Plotting.ProfilePlotter import RingPeakEnergyPlotter
from src.core.Validation.RingPeakEnergyAnalyzer import (
    RingPeakEnergyAnalyzer,
    SubdetectorDefinition,
)

matplotlib.use("Agg")


@pytest.fixture(autouse=True)
def _close_figures() -> None:
    yield
    plt.close("all")


def _build_synthetic_100_ring_profiles() -> tuple[np.ndarray, np.ndarray]:
    sig = np.zeros(100, dtype=np.float64)
    bkg = np.zeros(100, dtype=np.float64)

    sig[0:3] = [0.05, 0.03, 0.01]
    sig[3:8] = 0.002

    sig[8:11] = [0.20, 0.15, 0.05]
    sig[11:72] = 0.10 / 61.0

    sig[72:75] = [0.15, 0.10, 0.05]
    sig[75:80] = 0.008

    sig[80:83] = [0.02, 0.01, 0.01]
    sig[83:88] = 0.002

    sig[88:91] = [0.003, 0.002, 0.001]
    sig[91:100] = 0.004 / 9.0

    bkg[0:3] = [0.02, 0.02, 0.01]
    bkg[3:8] = 0.01

    bkg[8:11] = [0.10, 0.10, 0.05]
    bkg[11:72] = 0.15 / 61.0

    bkg[72:75] = [0.08, 0.07, 0.05]
    bkg[75:80] = 0.02

    bkg[80:83] = [0.03, 0.02, 0.01]
    bkg[83:88] = 0.008

    bkg[88:91] = [0.03, 0.02, 0.01]
    bkg[91:100] = 0.04 / 9.0

    return sig, bkg


def test_ring_index_resolution_matches_dataset_conventions(tmp_path: pathlib.Path) -> None:
    ds_50 = EgammaNpzDataset(drive_path=str(tmp_path), percentage=0.5)
    resolved_50 = RingPeakEnergyAnalyzer.resolve_ring_indexes(50)
    assert np.array_equal(resolved_50, np.asarray(ds_50.indexes))

    resolved_100 = RingPeakEnergyAnalyzer.resolve_ring_indexes(100)
    assert np.array_equal(resolved_100, np.arange(100))

    analyzer = RingPeakEnergyAnalyzer(top_k=3)
    pos_map_100 = analyzer.resolve_subdetector_positions(100)
    assert list(pos_map_100.keys()) == ["PreSampler", "EM1", "EM2", "EM3", "TileCal"]
    assert len(pos_map_100["PreSampler"]) == 8
    assert len(pos_map_100["EM1"]) == 64
    assert len(pos_map_100["EM2"]) == 8
    assert len(pos_map_100["EM3"]) == 8
    assert len(pos_map_100["TileCal"]) == 12

    pos_map_50 = analyzer.resolve_subdetector_positions(50, ring_indexes=resolved_50)
    assert [len(v) for v in pos_map_50.values()] == [4, 32, 4, 4, 6]

    mapping_df = analyzer.describe_subdetector_mapping(50, ring_indexes=resolved_50)
    assert list(mapping_df["subdetector"]) == ["PreSampler", "EM1", "EM2", "EM3", "TileCal"]
    assert list(mapping_df["n_peak_rings"]) == [3, 3, 3, 3, 3]
    assert mapping_df.loc[1, "peak_global_rings"] == (8, 9, 10)

    with pytest.raises(ValueError):
        RingPeakEnergyAnalyzer(top_k=0)
    with pytest.raises(ValueError):
        RingPeakEnergyAnalyzer(subdetectors=[])
    with pytest.raises(ValueError):
        RingPeakEnergyAnalyzer.resolve_ring_indexes(0)
    with pytest.raises(ValueError):
        RingPeakEnergyAnalyzer.resolve_ring_indexes(10, ring_indexes=np.arange(5))


def test_analyze_profile_and_signal_vs_background_math() -> None:
    sig, bkg = _build_synthetic_100_ring_profiles()
    assert np.isclose(np.sum(sig), 1.0)
    assert np.isclose(np.sum(bkg), 1.0)

    analyzer = RingPeakEnergyAnalyzer(top_k=3)
    comp_df = analyzer.analyze_signal_vs_background(
        mean_signal=sig,
        mean_bkg=bkg,
        region="et1_eta1",
    )

    assert list(comp_df["subdetector"]) == ["PreSampler", "EM1", "EM2", "EM3", "TileCal"]
    assert np.isclose(comp_df["signal_subdetector_share_pct"].sum(), 100.0)
    assert np.isclose(comp_df["background_subdetector_share_pct"].sum(), 100.0)

    ps_row = comp_df.loc[comp_df["subdetector"] == "PreSampler"].iloc[0]
    assert np.isclose(ps_row["signal_peak_energy"], 0.09)
    assert np.isclose(ps_row["signal_subdetector_energy"], 0.10)
    assert np.isclose(ps_row["signal_intra_detector_pct"], 90.0)
    assert np.isclose(ps_row["signal_global_calorimeter_pct"], 9.0)
    assert np.isclose(ps_row["signal_ring_1_intra_pct"], 50.0)
    assert np.isclose(ps_row["signal_ring_2_intra_pct"], 30.0)
    assert np.isclose(ps_row["signal_ring_3_intra_pct"], 10.0)
    assert np.isclose(ps_row["background_intra_detector_pct"], 50.0)
    assert np.isclose(ps_row["background_global_calorimeter_pct"], 5.0)
    assert np.isclose(ps_row["delta_intra_detector_pct"], 40.0)
    assert np.isclose(ps_row["delta_global_calorimeter_pct"], 4.0)

    em1_row = comp_df.loc[comp_df["subdetector"] == "EM1"].iloc[0]
    assert np.isclose(em1_row["signal_intra_detector_pct"], 80.0)
    assert np.isclose(em1_row["signal_global_calorimeter_pct"], 40.0)
    assert np.isclose(em1_row["background_intra_detector_pct"], 62.5)
    assert np.isclose(em1_row["background_global_calorimeter_pct"], 25.0)

    conc = analyzer.compute_total_peak_concentration(comp_df)
    assert conc["total_peak_rings"] == 15.0
    assert conc["total_rings"] == 100.0
    expected_sig_total_global = (0.09 + 0.40 + 0.30 + 0.04 + 0.006) * 100.0
    expected_bkg_total_global = (0.05 + 0.25 + 0.20 + 0.06 + 0.06) * 100.0
    assert np.isclose(conc["signal_total_peak_global_pct"], expected_sig_total_global)
    assert np.isclose(conc["background_total_peak_global_pct"], expected_bkg_total_global)
    assert np.isclose(
        conc["delta_total_peak_global_pct"],
        expected_sig_total_global - expected_bkg_total_global,
    )


def test_zero_denominator_and_50_ring_and_region_aggregation() -> None:
    analyzer = RingPeakEnergyAnalyzer(top_k=3)
    zero_profile = np.zeros(50, dtype=np.float64)
    zero_metrics = analyzer.analyze_profile(zero_profile, region="empty", class_name="Signal")
    assert all(m.intra_detector_pct == 0.0 for m in zero_metrics)
    assert all(m.global_calorimeter_pct == 0.0 for m in zero_metrics)
    assert all(m.to_dict()["subdetector"] == m.subdetector for m in zero_metrics)

    sig_100, bkg_100 = _build_synthetic_100_ring_profiles()
    df_r1 = analyzer.analyze_signal_vs_background(sig_100, bkg_100, region="et0_eta0")
    df_r2 = analyzer.analyze_signal_vs_background(sig_100 * 0.5, bkg_100 * 0.5, region="et0_eta1")
    df_r2["signal_intra_detector_pct"] = df_r2["signal_intra_detector_pct"] - 10.0

    combined = pd.concat([df_r1, df_r2], ignore_index=True)
    summary = analyzer.summarize_across_regions(combined)
    assert list(summary["subdetector"]) == ["PreSampler", "EM1", "EM2", "EM3", "TileCal"]
    assert all(summary["n_regions"] == 2)

    ps_sum = summary.loc[summary["subdetector"] == "PreSampler"].iloc[0]
    assert np.isclose(ps_sum["signal_intra_detector_pct_mean"], 85.0)
    assert np.isclose(ps_sum["signal_intra_detector_pct_min"], 80.0)
    assert np.isclose(ps_sum["signal_intra_detector_pct_max"], 90.0)
    assert np.isclose(ps_sum["signal_intra_detector_pct_std"], np.std([90.0, 80.0], ddof=1))

    empty_summary = analyzer.summarize_across_regions(pd.DataFrame())
    assert empty_summary.empty

    custom_analyzer = RingPeakEnergyAnalyzer(
        top_k=3,
        subdetectors=(SubdetectorDefinition("OnlyTwo", 0, 2, "#000000"),),
    )
    m_small = custom_analyzer.analyze_profile(np.array([0.6, 0.4]))
    assert m_small[0].n_peak_rings == 2
    assert np.isclose(m_small[0].intra_detector_pct, 100.0)


def test_variance_cv_and_hadronic_vs_em_comparison() -> None:
    sig, bkg = _build_synthetic_100_ring_profiles()
    std_sig = np.full(100, 0.002, dtype=np.float64)
    std_bkg = np.full(100, 0.002, dtype=np.float64)

    std_sig[0:3] = 0.01
    std_sig[88:100] = 0.001
    std_bkg[88:100] = 0.02

    analyzer = RingPeakEnergyAnalyzer(top_k=3)
    with pytest.raises(ValueError):
        analyzer.analyze_profile(sig, std_profile=np.ones(10))

    comp_df = analyzer.analyze_signal_vs_background(
        mean_signal=sig,
        mean_bkg=bkg,
        region="et2_eta1",
        std_signal=std_sig,
        std_bkg=std_bkg,
    )

    assert np.isclose(comp_df["signal_variance_share_pct"].sum(), 100.0)
    assert np.isclose(comp_df["background_variance_share_pct"].sum(), 100.0)

    ps_row = comp_df.loc[comp_df["subdetector"] == "PreSampler"].iloc[0]
    expected_ps_var = float(np.mean(std_sig[0:8] ** 2))
    expected_ps_std = float(np.mean(std_sig[0:8]))
    expected_ps_mean_e = 0.10 / 8.0
    assert np.isclose(ps_row["signal_mean_ring_energy"], expected_ps_mean_e)
    assert np.isclose(ps_row["signal_mean_ring_variance"], expected_ps_var)
    assert np.isclose(ps_row["signal_mean_ring_std"], expected_ps_std)
    assert np.isclose(ps_row["signal_subdetector_cv"], expected_ps_std / expected_ps_mean_e)
    assert np.isclose(ps_row["signal_peak_mean_ring_variance"], 0.01**2)
    assert np.isclose(ps_row["signal_peak_cv"], 0.01 / 0.03)

    tile_row = comp_df.loc[comp_df["subdetector"] == "TileCal"].iloc[0]
    assert np.isclose(tile_row["background_mean_ring_variance"], 0.02**2)
    assert np.isclose(tile_row["ratio_bkg_to_sig_mean_variance"], (0.02**2) / (0.001**2))
    assert np.isclose(tile_row["ratio_bkg_to_sig_peak_variance"], (0.02**2) / (0.001**2))
    assert tile_row["ratio_bkg_to_sig_cv"] > 0.0

    had_summary = analyzer.compare_hadronic_vs_em_variance(comp_df)
    assert had_summary["hadronic_subdetector"] == "TileCal"
    assert had_summary["background_highest_abs_variance_subdetector"] == "TileCal"
    assert had_summary["background_hadronic_to_em_variance_ratio"] > 1.0
    assert had_summary["signal_hadronic_cv"] > had_summary["signal_em_mean_cv"]
    assert had_summary["signal_highest_cv_subdetector"] == "TileCal"

    with pytest.raises(ValueError):
        analyzer.compare_hadronic_vs_em_variance(pd.DataFrame())
    with pytest.raises(ValueError):
        analyzer.compare_hadronic_vs_em_variance(comp_df, hadronic_subdetector="Missing")

    regional_var = analyzer.summarize_across_regions(comp_df)
    assert "signal_mean_ring_variance_mean" in regional_var.columns
    assert "background_subdetector_cv_mean" in regional_var.columns


def test_ring_peak_energy_plotter_bilingual_outputs(tmp_path: pathlib.Path) -> None:
    sig_100, bkg_100 = _build_synthetic_100_ring_profiles()
    std_sig_100 = np.full(100, 0.002, dtype=np.float64)
    std_bkg_100 = np.full(100, 0.004, dtype=np.float64)

    analyzer = RingPeakEnergyAnalyzer(top_k=3)
    df_r1 = analyzer.analyze_signal_vs_background(
        sig_100,
        bkg_100,
        region="et0_eta0",
        std_signal=std_sig_100,
        std_bkg=std_bkg_100,
    )
    df_r2 = analyzer.analyze_signal_vs_background(
        sig_100,
        bkg_100,
        region="et1_eta1",
        std_signal=std_sig_100 * 1.2,
        std_bkg=std_bkg_100 * 1.3,
    )
    regional_df = pd.concat([df_r1, df_r2], ignore_index=True)

    rng = np.random.default_rng(42)
    ctx_data = rng.uniform(0.01, 1.0, size=(20, 100))
    ctx_labels = np.array([1] * 10 + [0] * 10)
    et_gev = rng.uniform(15.0, 60.0, size=20)
    hist2d_data = analyzer.compute_subdetector_et_histogram2d(
        rings_data=ctx_data,
        transverse_energy_gev=et_gev,
        labels=ctx_labels,
        n_et_bins=6,
        et_range=(15.0, 60.0),
    )

    for lang in ("en", "pt"):
        lang_dir = tmp_path / lang
        plotter = RingPeakEnergyPlotter(language=lang, top_k=3)

        p_prof = plotter.plot_highlighted_peak_profile(
            mean_signal=sig_100,
            std_signal=std_sig_100,
            mean_bkg=bkg_100,
            std_bkg=std_bkg_100,
            output_dir=lang_dir,
            filename="global_peak_profile.png",
            scope_label="Global",
            language=lang,
        )
        p_bars = plotter.plot_subdetector_peak_bars(
            comparison_df=df_r1,
            output_dir=lang_dir,
            filename="global_peak_bars.pdf",
            scope_label="Global",
            language=lang,
        )
        p_var_bars = plotter.plot_subdetector_variance_bars(
            comparison_df=df_r1,
            output_dir=lang_dir,
            filename="global_variance_bars.png",
            scope_label="Global",
            language=lang,
        )
        p_heat_intra = plotter.plot_regional_peak_heatmap(
            regional_df=regional_df,
            output_dir=lang_dir,
            filename="heatmap_intra.png",
            metric="intra_detector_pct",
            language=lang,
        )
        p_heat_global = plotter.plot_regional_peak_heatmap(
            regional_df=regional_df,
            output_dir=lang_dir,
            filename="heatmap_global.pdf",
            metric="global_calorimeter_pct",
            language=lang,
        )
        p_heat_var = plotter.plot_regional_peak_heatmap(
            regional_df=regional_df,
            output_dir=lang_dir,
            filename="heatmap_variance.png",
            metric="mean_ring_variance",
            language=lang,
        )
        p_heat_cv = plotter.plot_regional_peak_heatmap(
            regional_df=regional_df,
            output_dir=lang_dir,
            filename="heatmap_cv.pdf",
            metric="subdetector_cv",
            language=lang,
        )
        p_hist2d = plotter.plot_subdetector_et_hist2d(
            hist2d_data=hist2d_data,
            output_dir=lang_dir,
            filename="et_vs_subdetector_hist2d.png",
            scope_label="et1_eta2",
            et_label="trig_L2_calo_et",
            language=lang,
        )

        ctx = RegionPlotContext(
            iet=1,
            ieta=2,
            output_dir=str(lang_dir),
            data=ctx_data,
            target=ctx_labels,
            language=lang,
        )
        p_ctx = plotter.plot(ctx, x_rings=ctx_data)

        for saved in (
            p_prof,
            p_bars,
            p_var_bars,
            p_heat_intra,
            p_heat_global,
            p_heat_var,
            p_heat_cv,
            p_hist2d,
            p_ctx,
        ):
            path_obj = pathlib.Path(saved)
            assert path_obj.exists() and path_obj.stat().st_size > 0

    with pytest.raises(ValueError):
        RingPeakEnergyPlotter(language="en").plot_regional_peak_heatmap(
            regional_df=regional_df,
            output_dir=tmp_path,
            filename="bad.png",
            metric="unknown_metric",
        )


def test_subdetector_et_histogram2d_computation() -> None:
    sig_100, bkg_100 = _build_synthetic_100_ring_profiles()
    rings = np.vstack([np.tile(sig_100, (4, 1)), np.tile(bkg_100, (3, 1))])
    et_gev = np.array([25.0, 25.0, 45.0, 45.0, 30.0, 30.0, 50.0])
    labels = np.array([1, 1, 1, 1, 0, 0, 0])

    analyzer = RingPeakEnergyAnalyzer(top_k=3)
    res = analyzer.compute_subdetector_et_histogram2d(
        rings_data=rings,
        transverse_energy_gev=et_gev,
        labels=labels,
        n_et_bins=3,
        et_range=(20.0, 50.0),
    )

    assert res["subdetectors"] == ["PreSampler", "EM1", "EM2", "EM3", "TileCal"]
    assert res["n_signal"] == 4
    assert res["n_background"] == 3
    assert res["n_rings"] == 100
    assert res["signal_hist2d"].shape == (3, 5)
    assert res["background_hist2d"].shape == (3, 5)
    assert np.isclose(np.sum(res["signal_hist2d"]), 4.0)
    assert np.isclose(np.sum(res["background_hist2d"]), 3.0)

    with pytest.raises(ValueError):
        analyzer.compute_subdetector_et_histogram2d(rings, et_gev[:-1], labels)
    with pytest.raises(ValueError):
        analyzer.compute_subdetector_et_histogram2d(rings, et_gev, labels, n_et_bins=0)
