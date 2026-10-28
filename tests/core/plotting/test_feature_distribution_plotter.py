import numpy as np
import pytest

from src.core.Plotting.FeatureDistributionPlotter import FeatureDistributionPlotter


@pytest.fixture
def sample_feature_plot_data():
    bin_centers = np.linspace(0.0, 1.0, 10)
    true_counts = np.array([100, 120, 150, 180, 200, 190, 160, 130, 90, 50])
    model_1_counts = np.array([95, 115, 142, 175, 195, 185, 155, 125, 88, 48])
    model_2_counts = np.array([90, 110, 135, 168, 190, 180, 150, 120, 85, 45])

    models = [
        {
            "name": "Model 1",
            "counts": model_1_counts,
            "errors": np.sqrt(model_1_counts),
            "is_ml": True,
            "color": "tab:blue",
            "marker": "o",
            "label": "Neural Model 1",
        },
        {
            "name": "Athena Cut",
            "counts": model_2_counts,
            "errors": np.sqrt(model_2_counts),
            "is_ml": False,
            "color": "black",
            "marker": "s",
            "label": "Athena Tight Cut",
        },
    ]

    return {
        "Reta": {
            "display_title": r"$R_{\eta}$",
            "bin_centers": bin_centers,
            "true_counts": true_counts,
            "models": models,
        },
        "Rhad": {
            "display_title": r"$R_{\text{had}}$",
            "bin_centers": bin_centers,
            "true_counts": true_counts,
            "models": models,
        },
    }


def test_plot_single_feature(tmp_path, sample_feature_plot_data):
    plotter = FeatureDistributionPlotter()
    feat_data = sample_feature_plot_data["Reta"]
    output_file = tmp_path / "single_feature.png"

    fig = plotter.plot_single_feature(
        feature_data=feat_data,
        feature_name="Reta",
        region_key="iet1.ieta1",
        is_signal=True,
        log_y=True,
        output_path=output_file,
        plot_format="png",
    )

    assert fig is not None
    assert output_file.exists()


def test_plot_overview_grid(sample_feature_plot_data):
    plotter = FeatureDistributionPlotter()
    features = ["Reta", "Rhad"]

    fig = plotter.plot_overview_grid(
        plot_data=sample_feature_plot_data,
        features_to_plot=features,
        region_key="iet1.ieta1",
        is_signal=True,
        log_y=False,
    )

    assert fig is not None


def test_plot_and_export_all(tmp_path, sample_feature_plot_data):
    plotter = FeatureDistributionPlotter()
    features = ["Reta", "Rhad"]
    export_dir = tmp_path / "plots"

    fig = plotter.plot_and_export_all(
        plot_data=sample_feature_plot_data,
        features_to_plot=features,
        output_dir=export_dir,
        region_key="iet1.ieta1",
        is_signal=False,
        plot_format="png",
    )

    assert fig is not None
    assert (export_dir / "Reta_background_selection.png").exists()
    assert (export_dir / "Rhad_background_selection.png").exists()

