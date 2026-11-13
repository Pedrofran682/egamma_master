import pathlib
import numpy as np
import pytest
import torch
import torch.nn as nn

from src.core.Plotting.PileupPlotter import PileupPlotter
from src.core.Validation.HoldoutEvaluator import HoldoutEvaluator
from src.core.Validation.PileupThresholdCalibrator import (
    OperatingPointCalibration,
    Pileup2DDistribution,
    PileupThresholdCalibrator,
)


class DummyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.fc(x))


def test_pileup_2d_distribution_computation() -> None:
    calibrator = PileupThresholdCalibrator(bins_mu=10)
    np.random.seed(42)
    mu = np.random.uniform(20.0, 60.0, size=500)
    disc = 0.05 * mu + np.random.normal(0.0, 1.0, size=500)

    dist = calibrator.compute_2d_distribution(mu, disc, bins=20)
    assert isinstance(dist, Pileup2DDistribution)
    assert dist.hist2d_counts.shape == (20, 20)
    assert len(dist.hist2d_xedges) == 21
    assert len(dist.hist2d_yedges) == 21
    assert len(dist.mu_values) == 500


def test_calibrate_working_point_linear_fit() -> None:
    calibrator = PileupThresholdCalibrator(bins_mu=8, n_bootstraps=20, random_seed=42)
    np.random.seed(42)
    mu = np.random.uniform(20.0, 60.0, size=2000)
    true_slope = 0.04
    true_intercept = 1.2
    disc = true_slope * mu + true_intercept + np.random.normal(0.0, 0.2, size=2000)

    cal = calibrator.calibrate_working_point(mu, disc, target_pd=0.80, working_point_name="Tight")
    assert isinstance(cal, OperatingPointCalibration)
    assert cal.working_point_name == "Tight"
    assert cal.target_pd == 0.80
    assert len(cal.mu_bin_centers) >= 5
    assert len(cal.slice_thresholds) == len(cal.mu_bin_centers)
    assert len(cal.slice_threshold_errors) == len(cal.mu_bin_centers)
    assert pytest.approx(cal.slope, abs=0.02) == true_slope
    assert cal.r_squared > 0.85


def test_calibrate_multiple_points() -> None:
    calibrator = PileupThresholdCalibrator(bins_mu=6, n_bootstraps=10)
    np.random.seed(42)
    mu = np.random.uniform(15.0, 55.0, size=1500)
    disc = 0.03 * mu + np.random.normal(0.0, 0.5, size=1500)

    working_points = {
        "Tight": 0.80,
        "Medium": 0.90,
        "Loose": 0.95,
    }

    results = calibrator.calibrate_multiple_points(mu, disc, working_points)
    assert set(results.keys()) == {"Tight", "Medium", "Loose"}

    for wp_name, cal in results.items():
        assert isinstance(cal, OperatingPointCalibration)
        assert cal.target_pd == working_points[wp_name]
        assert cal.slope != 0.0


def test_apply_threshold() -> None:
    mu = np.array([20.0, 40.0, 60.0])
    slope = 0.1
    intercept = 1.0

    thresholds = slope * mu + intercept
    disc = np.array([thresholds[0] + 0.1, thresholds[1] - 0.1, thresholds[2]])

    preds = PileupThresholdCalibrator.apply_threshold(disc, mu, slope, intercept)
    np.testing.assert_array_equal(preds, np.array([1, 0, 1]))


def test_calibrator_invalid_target_pd() -> None:
    calibrator = PileupThresholdCalibrator()
    mu = np.array([1.0, 2.0])
    disc = np.array([1.0, 2.0])
    with pytest.raises(ValueError, match="target_pd must be in range"):
        calibrator.calibrate_working_point(mu, disc, target_pd=1.5)


def test_holdout_evaluator_logits() -> None:
    model = DummyModel()
    features = np.random.randn(50, 4).astype(np.float32)
    target = np.random.randint(0, 2, size=50)
    indices = np.arange(50)
    ring_indices = np.arange(4)
    device = torch.device("cpu")

    evaluator = HoldoutEvaluator(default_target_pd=0.85)
    result = evaluator.evaluate(
        model=model,
        data=features,
        target=target,
        test_indices=indices,
        ring_column_indices=ring_indices,
        device=device,
    )

    assert result.predicted_logits is not None
    assert len(result.predicted_logits) == 50
    assert np.all(np.isfinite(result.logits))

    direct_logits = HoldoutEvaluator.predict_logits(model, features, device)
    assert len(direct_logits) == 50
    np.testing.assert_allclose(result.logits, direct_logits, rtol=1e-5)


def test_pileup_plotter_2d_calibration(tmp_path: pathlib.Path) -> None:
    plotter = PileupPlotter()
    np.random.seed(42)
    mu = np.random.uniform(20.0, 60.0, size=500)
    disc = 0.04 * mu + np.random.normal(0.0, 0.5, size=500)

    calibrator = PileupThresholdCalibrator(bins_mu=6, n_bootstraps=5)
    dist = calibrator.compute_2d_distribution(mu, disc, bins=15)
    cal = calibrator.calibrate_working_point(mu, disc, target_pd=0.80, working_point_name="Tight")

    out_file = tmp_path / "test_2d_calib.png"
    fig = plotter.plot_2d_threshold_calibration(
        distribution=dist,
        calibrations=[cal],
        title="Test Calibration",
        region_label="iet1.ieta1",
        output_path=out_file,
        file_format="png",
    )

    assert fig is not None
    assert out_file.exists()

