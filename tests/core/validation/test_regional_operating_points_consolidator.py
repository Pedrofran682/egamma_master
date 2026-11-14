from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn

from src.core.Validation.HoldoutEvaluator import HoldoutEvaluator
from src.core.Validation.RegionalOperatingPointsConsolidator import (
    RegionalOperatingPointsConsolidator,
)
from src.core.Validation.ResultAggregator import ResultAggregator


@pytest.fixture
def mock_consolidator(tmp_path: Path) -> RegionalOperatingPointsConsolidator:
    class DummyConsolidator(RegionalOperatingPointsConsolidator):
        def __init__(self) -> None:
            self.target_pd = 0.9424
            self.working_point = "loose"
            self.method_aliases = {
                "calibrated_hard_cut": "Calibrated Hard Cut (Target PD: 94.24%)",
                "pileup_corrected_cut": "Pileup-Corrected Cut T(mu) (Target PD)",
                "athena_cut": "Athena Loose Cut",
                "training_best_sp": "Training Best SP",
            }
            self._records = []
            self._training_records = []
            self._athena_matched_records = []
            athena_aliases = self._athena_matched_method_aliases()
            for et in range(2):
                for eta in range(2):
                    self._training_records.append(
                        {
                            "iet": et,
                            "ieta": eta,
                            "method_key": self.TRAINING_METHOD_KEY,
                            "method": self.method_aliases[self.TRAINING_METHOD_KEY],
                            "repeat": 0,
                            "fold": 1,
                            "pd": 0.9612,
                            "pd_uncertainty": 0.0045,
                            "sp": 0.9550,
                            "sp_uncertainty": 0.0032,
                            "pf": 0.0510,
                            "pf_uncertainty": 0.0021,
                            "mean_rep_sp": 0.9520,
                            "std_rep_sp": 0.0020,
                        }
                    )
                    self._athena_matched_records.append(self._training_records[-1])
                    for m_key in self.DEFAULT_METHOD_KEYS:
                        sp_val = 0.9580 if m_key == "pileup_corrected_cut" else 0.9410
                        self._athena_matched_records.append(
                            {
                                "iet": et,
                                "ieta": eta,
                                "method_key": m_key,
                                "method": athena_aliases[m_key],
                                "target_pd": 0.9350,
                                "decision_threshold": "0.1000",
                                "pd": 0.9350,
                                "pd_uncertainty": 0.0030,
                                "sp": sp_val,
                                "sp_uncertainty": 0.0028,
                                "pf": 0.0420,
                                "pf_uncertainty": 0.0019,
                                "eff": 0.9350,
                                "eff_uncertainty": 0.0025,
                                "bg_eff": 0.9580,
                                "bg_eff_uncertainty": 0.0019,
                                "tp": 935,
                                "fp": 42,
                                "tn": 958,
                                "fn": 65,
                                "total": 2000,
                            }
                        )
                        self._records.append(
                            {
                                "iet": et,
                                "ieta": eta,
                                "method_key": m_key,
                                "method": self.method_aliases[m_key],
                                "target_pd": 0.9424,
                                "decision_threshold": "0.1000",
                                "pd": 0.9424,
                                "pd_uncertainty": 0.0035,
                                "sp": 0.9500 if m_key == "pileup_corrected_cut" else 0.9400,
                                "sp_uncertainty": 0.0030,
                                "pf": 0.0500,
                                "pf_uncertainty": 0.0020,
                                "eff": 0.9400,
                                "eff_uncertainty": 0.0025,
                                "bg_eff": 0.9500,
                                "bg_eff_uncertainty": 0.0020,
                                "tp": 942,
                                "fp": 50,
                                "tn": 950,
                                "fn": 58,
                                "total": 2000,
                            }
                        )

    return DummyConsolidator()


def test_build_tidy_dataframe(mock_consolidator: RegionalOperatingPointsConsolidator) -> None:
    df = mock_consolidator.build_tidy_dataframe()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2 * 2 * 3
    assert "iet" in df.columns
    assert "ieta" in df.columns
    assert "method_key" in df.columns
    assert "sp" in df.columns


def test_build_grid_table(mock_consolidator: RegionalOperatingPointsConsolidator) -> None:
    grid = mock_consolidator.build_grid_table(as_percentage=True, precision=2)
    assert isinstance(grid.columns, pd.MultiIndex)
    assert isinstance(grid.index, pd.MultiIndex)
    assert list(grid.index.names) == ["eta", "Method"]
    assert len(grid.index) == 2 * 3
    assert len(grid.columns) == 2 * 3


def test_to_latex_snippet(mock_consolidator: RegionalOperatingPointsConsolidator) -> None:
    latex = mock_consolidator.to_latex_snippet(as_percentage=True, precision=2)
    assert "\\begin{table}" in latex
    assert "\\resizebox{\\textwidth}{!}" in latex
    assert "\\multirow{3}{*}{$0.00 \\le |\\eta| < 0.80$}" in latex
    assert "$15 \\le E_T < 20\\,\\text{GeV}$" in latex
    assert "Method" in latex
    assert "Método" not in latex
    assert "$P_D[\\%]$" in latex
    assert "\\end{tabular}" in latex
    assert "\\end{table}" in latex


def test_export_parquet_and_latex(
    mock_consolidator: RegionalOperatingPointsConsolidator, tmp_path: Path
) -> None:
    parquet_path = tmp_path / "table.parquet"
    latex_path = tmp_path / "table.tex"

    saved_parquet = mock_consolidator.export_parquet(parquet_path)
    assert saved_parquet.exists()

    saved_latex = mock_consolidator.export_latex(latex_path)
    assert saved_latex.exists()
    content = saved_latex.read_text(encoding="utf-8")
    assert "\\begin{table}" in content


def test_result_aggregator_streaming_no_cache(tmp_path: Path) -> None:
    df_reg0 = pd.DataFrame(
        [
            {
                "repeat": 0,
                "fold": 0,
                "best_sp_value": 0.91,
                "best_pd_value": 0.92,
                "best_fa_value": 0.08,
                "best_weights": {"w": 1},
            },
            {
                "repeat": 0,
                "fold": 1,
                "best_sp_value": 0.95,
                "best_pd_value": 0.96,
                "best_fa_value": 0.04,
                "best_weights": {"w": 2},
            },
        ]
    )
    df_reg1 = pd.DataFrame(
        [
            {
                "repeat": 0,
                "fold": 0,
                "best_sp_value": 0.88,
                "best_pd_value": 0.90,
                "best_fa_value": 0.10,
                "best_weights": {"w": 3},
            },
        ]
    )
    df_reg0.to_pickle(tmp_path / "results.iet0.ieta0.pkl")
    df_reg1.to_pickle(tmp_path / "results.iet1.ieta2.pkl")

    agg = ResultAggregator(tmp_path)
    assert agg.has_region(0, 0) is True
    assert agg.has_region(1, 2) is True
    assert agg.has_region(4, 4) is False
    assert agg._grouped_dfs is None

    best_model, best_rep, mean_sp, _ = agg.get_best_model_for_region(0, 0, cache=False)
    assert agg._grouped_dfs is None
    assert best_rep == 0
    assert best_model["best_weights"] == {"w": 2}
    assert np.isclose(mean_sp, 0.93)

    metrics = agg.get_best_training_metrics_for_region(0, 0, cache=False)
    assert agg._grouped_dfs is None
    assert np.isclose(metrics["sp"], 0.95)
    assert np.isclose(metrics["pd"], 0.96)
    assert np.isclose(metrics["pf"], 0.04)
    assert np.isclose(metrics["sp_uncertainty"], np.std([0.91, 0.95], ddof=1))
    assert np.isclose(metrics["pd_uncertainty"], np.std([0.92, 0.96], ddof=1))
    assert np.isclose(metrics["pf_uncertainty"], np.std([0.08, 0.04], ddof=1))

    single_fold_metrics = agg.get_best_training_metrics_for_region(1, 2, cache=False)
    assert single_fold_metrics["sp_uncertainty"] == 0.0


def test_holdout_evaluator_batched_matches_unbatched() -> None:
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(4, 1), nn.Sigmoid())
    rng = np.random.default_rng(7)
    data = rng.normal(size=(15, 6)).astype(np.float32)
    target = np.array([1] * 8 + [0] * 7)
    test_idx = np.arange(15)
    ring_cols = np.array([0, 1, 2, 3])

    eval_full = HoldoutEvaluator(default_target_pd=0.9, batch_size=100)
    eval_chunked = HoldoutEvaluator(default_target_pd=0.9, batch_size=4)

    res_full = eval_full.evaluate(model, data, target, test_idx, ring_cols, torch.device("cpu"))
    res_chunked = eval_chunked.evaluate(model, data, target, test_idx, ring_cols, torch.device("cpu"))

    assert np.allclose(res_full.predicted_probabilities, res_chunked.predicted_probabilities)
    assert np.allclose(res_full.predicted_logits, res_chunked.predicted_logits)

    logits_full = HoldoutEvaluator.predict_logits(model, data[:, ring_cols], torch.device("cpu"), batch_size=100)
    logits_chunked = HoldoutEvaluator.predict_logits(model, data[:, ring_cols], torch.device("cpu"), batch_size=4)
    assert np.allclose(logits_full, logits_chunked)


def test_build_training_grid_table_and_latex(
    mock_consolidator: RegionalOperatingPointsConsolidator,
) -> None:
    grid = mock_consolidator.build_training_grid_table(as_percentage=True, precision=2)
    assert len(grid.index) == 2
    assert len(grid.columns) == 2 * 3
    assert "$15 \\le E_T < 20\\,\\text{GeV}$" in grid.columns.get_level_values(0)
    assert "$0.00 \\le |\\eta| < 0.80$" in grid.index.get_level_values(0)
    assert "$96.12 \\pm 0.45$" in grid.iloc[0, 0]

    latex = mock_consolidator.to_latex_snippet(grid_table=grid, as_percentage=True, precision=2)
    assert "Training Best SP" in latex
    assert "$0.00 \\le |\\eta| < 0.80$" in latex
    assert "$20 \\le E_T < 30\\,\\text{GeV}$" in latex


def test_build_athena_matched_comparison_grid_table_and_latex(
    mock_consolidator: RegionalOperatingPointsConsolidator,
) -> None:
    grid = mock_consolidator.build_athena_matched_comparison_grid_table(
        as_percentage=True, precision=2, bold_best_sp=True
    )
    assert len(grid.index) == 2 * 4
    assert len(grid.columns) == 2 * 3
    methods_in_index = list(grid.index.get_level_values(1))
    assert "Training Best SP" in methods_in_index
    assert "Calibrated Hard Cut (Athena Loose PD)" in methods_in_index
    assert "Pileup-Corrected Cut T(mu) (Athena Loose PD)" in methods_in_index
    assert "Athena Loose Cut" in methods_in_index
    assert "\\pm" in grid.iloc[0, 0]
    assert "\\mathbf{" in grid.iloc[2, 1]

    latex = mock_consolidator.to_latex_snippet(grid_table=grid, as_percentage=True, precision=2)
    assert "\\multirow{4}{*}{$0.00 \\le |\\eta| < 0.80$}" in latex
    assert "Calibrated Hard Cut (Athena Loose PD)" in latex
    assert "Pileup-Corrected Cut T(mu) (Athena Loose PD)" in latex
