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

