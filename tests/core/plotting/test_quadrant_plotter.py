import pathlib
import numpy as np
import pytest

from src.core.Plotting.QuadrantPlotter import QuadrantPlotter
from src.core.Validation.QuadrantAnalyzer import (
    QuadrantMetrics,
    RegionalQuadrantResult,
    StrategyMetrics,
)


@pytest.fixture
def mock_regional_result() -> RegionalQuadrantResult:
    np.random.seed(42)
    n_samples = 200
    labels = np.random.choice([0, 1], size=n_samples, p=[0.5, 0.5])
    preds1 = labels.copy()
    preds2 = labels.copy()

    # Flip 20 predictions randomly
    flip_idx1 = np.random.choice(n_samples, size=20, replace=False)
    preds1[flip_idx1] = 1 - preds1[flip_idx1]

    flip_idx2 = np.random.choice(n_samples, size=25, replace=False)
    preds2[flip_idx2] = 1 - preds2[flip_idx2]

    showershapes = {
        "reta": np.random.normal(loc=0.95, scale=0.05, size=n_samples),
        "rhad": np.random.exponential(scale=0.02, size=n_samples),
    }

    dummy_sm1 = StrategyMetrics.from_predictions(preds1, labels)
    dummy_sm2 = StrategyMetrics.from_predictions(preds2, labels)
    dummy_qm = QuadrantMetrics(
        both_correct=160,
        model1_only_correct=15,
        model2_only_correct=10,
        both_wrong=15,
        total_events=200,
    )

    return RegionalQuadrantResult(
        iet=1,
        ieta=1,
        strategy1_name="ModelV1",
        strategy2_name="AthenaTight",
        metrics_strategy1=dummy_sm1,
        metrics_strategy2=dummy_sm2,
        overall_metrics=dummy_qm,
        signal_metrics=dummy_qm,
        background_metrics=dummy_qm,
        preds_strategy1=preds1,
        preds_strategy2=preds2,
        labels=labels,
        threshold_strategy1=0.5,
        threshold_strategy2=0.5,
        showershapes=showershapes,
        et=np.random.uniform(20.0, 50.0, size=n_samples),
        eta=np.random.uniform(0.0, 1.37, size=n_samples),
    )


def test_plot_showershape_histogram_linear(tmp_path: pathlib.Path, mock_regional_result: RegionalQuadrantResult) -> None:
    plotter = QuadrantPlotter()
    output_path = plotter.plot_showershape_histogram(
        result=mock_regional_result,
        var_name="reta",
        output_dir=tmp_path,
        file_format="png",
        log_y=False,
    )

    assert output_path is not None
    assert pathlib.Path(output_path).exists()
    assert output_path.endswith("_reta_quadrant_hist.png")


def test_plot_showershape_histogram_log(tmp_path: pathlib.Path, mock_regional_result: RegionalQuadrantResult) -> None:
    plotter = QuadrantPlotter()
    output_path = plotter.plot_showershape_histogram(
        result=mock_regional_result,
        var_name="rhad",
        output_dir=tmp_path,
        file_format="png",
        log_y=True,
    )

    assert output_path is not None
    assert pathlib.Path(output_path).exists()
    assert output_path.endswith("_rhad_quadrant_hist.png")


def test_plot_workflow_with_log_y(tmp_path: pathlib.Path, mock_regional_result: RegionalQuadrantResult) -> None:
    plotter = QuadrantPlotter()
    saved = plotter.plot(
        results=[mock_regional_result],
        output_dir=tmp_path,
        file_format="png",
        log_y=True,
    )

    assert len(saved["histograms"]) == 2
    for p in saved["histograms"]:
        assert pathlib.Path(p).exists()
