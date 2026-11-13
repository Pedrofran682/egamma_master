import numpy as np
import pytest

from src.core.Plotting.PileupPlotter import PileupPlotter
from src.core.Validation.PileupAnalyzer import PileupAnalyzer


def test_pileup_analyzer_feature_extraction():
    feature_names = np.array(["avgmu", "ph_nGoodVtx", "trig_L2_calo_emaxs1", "trig_L2_calo_e2tsts1"])
    features = np.array([
        [30.0, 15.0, 100.0, 50.0],
        [45.0, 22.0, 80.0, 20.0],
    ])

    mu = PileupAnalyzer.extract_feature(features, feature_names, "avgmu")
    assert np.allclose(mu, [30.0, 45.0])

    eratio = PileupAnalyzer.extract_feature(features, feature_names, "Eratio")
    expected_eratio = np.array([(100.0 - 50.0) / 150.0, (80.0 - 20.0) / 100.0])
    assert np.allclose(eratio, expected_eratio)

    with pytest.raises(KeyError):
        PileupAnalyzer.extract_feature(features, feature_names, "non_existent_feature")


def test_pileup_analyzer_efficiency_vs_mu():
    rng = np.random.default_rng(42)
    n = 500
    mu = rng.uniform(10.0, 60.0, size=n)
    y_true = np.ones(n, dtype=int)
    predictions = rng.binomial(1, 0.9, size=n)

    result = PileupAnalyzer.compute_efficiency_vs_mu(
        mu_values=mu,
        y_true=y_true,
        predictions=predictions,
        bins=5,
        is_signal=True,
    )

    assert "bin_centers" in result
    assert "efficiencies" in result
    assert len(result["bin_centers"]) == 5
    assert np.all((result["efficiencies"] >= 0.0) & (result["efficiencies"] <= 1.0))
    assert np.all(result["uncertainties"] >= 0.0)
    assert np.sum(result["total_counts"]) == n


def test_pileup_analyzer_profile_vs_mu():
    rng = np.random.default_rng(42)
    n = 300
    mu = rng.uniform(10.0, 50.0, size=n)
    obs = 0.5 * mu + rng.normal(0.0, 1.0, size=n)

    result = PileupAnalyzer.compute_profile_vs_mu(
        mu_values=mu,
        observable_values=obs,
        bins=5,
    )

    assert "bin_centers" in result
    assert "means" in result
    assert "sems" in result
    assert len(result["means"]) == 5
    assert np.sum(result["counts"]) == n


def test_pileup_analyzer_vertex_correlation():
    rng = np.random.default_rng(42)
    n = 400
    mu = rng.uniform(15.0, 65.0, size=n)
    npv = 0.7 * mu + rng.normal(2.0, 1.0, size=n)

    result = PileupAnalyzer.compute_vertex_correlation(
        mu_values=mu,
        npv_values=npv,
        bins=20,
    )

    assert "slope" in result
    assert "intercept" in result
    assert "r_squared" in result
    assert result["r_squared"] > 0.8
    assert "profile" in result
    assert "hist2d_counts" in result


def test_pileup_plotter_rendering(tmp_path):
    plotter = PileupPlotter()

    centers = np.linspace(20.0, 50.0, 5)
    eff_1 = np.array([0.95, 0.94, 0.93, 0.92, 0.90])
    err_1 = np.array([0.01, 0.01, 0.01, 0.01, 0.02])

    models_data = [
        {
            "label": "Model 1",
            "bin_centers": centers,
            "efficiencies": eff_1,
            "uncertainties": err_1,
            "color": "tab:blue",
            "marker": "o",
            "is_ml": True,
            "is_reference": True,
        }
    ]

    fig_eff = plotter.plot_efficiency_vs_mu(
        models_data=models_data,
        region_label="iet1.ieta1",
        output_path=tmp_path / "eff_vs_mu.png",
        file_format="png",
    )
    assert fig_eff is not None

    profiles_data = [
        {
            "label": "Signal Reta",
            "bin_centers": centers,
            "means": np.array([0.92, 0.91, 0.90, 0.89, 0.88]),
            "sems": np.array([0.005, 0.005, 0.005, 0.005, 0.005]),
            "color": "tab:blue",
            "marker": "o",
        }
    ]
    fig_prof = plotter.plot_profile_vs_mu(
        profiles_data=profiles_data,
        observable_label=r"$R_{\eta}$",
        region_label="iet1.ieta1",
        output_path=tmp_path / "profile_vs_mu.png",
        file_format="png",
    )
    assert fig_prof is not None

    corr_res = PileupAnalyzer.compute_vertex_correlation(
        mu_values=centers * 1.5,
        npv_values=centers * 0.8,
        bins=10,
    )
    fig_corr = plotter.plot_vertex_correlation(
        correlation_result=corr_res,
        region_label="iet1.ieta1",
        output_path=tmp_path / "corr_vs_mu.png",
        file_format="png",
    )
    assert fig_corr is not None

