from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.utils import (
    compute_ks_test_with_uncertainty,
    compute_wasserstein_with_uncertainty,
    extract_feature_values,
    save_table,
)


def test_save_table(tmp_path: Path):
    df = pd.DataFrame({"col_a": [1, 2], "col_b": [3, 4]})
    out_file = save_table(df, tmp_path, "et1_eta1", "test_metrics")

    assert out_file.exists()
    assert out_file.name == "et1_eta1_test_metrics.csv"
    reloaded = pd.read_csv(out_file)
    assert len(reloaded) == 2
    assert list(reloaded.columns) == ["col_a", "col_b"]


def test_extract_feature_values():
    features = np.array([
        [10.0, 50.0, 20.0],
        [20.0, 70.0, 10.0],
    ])
    names = np.array(["avgmu", "trig_L2_calo_emaxs1", "trig_L2_calo_e2tsts1"])

    mu = extract_feature_values(features, names, "avgmu")
    assert np.allclose(mu, [10.0, 20.0])

    eratio = extract_feature_values(features, names, "Eratio")
    expected = np.array([(50.0 - 20.0) / 70.0, (70.0 - 10.0) / 80.0])
    assert np.allclose(eratio, expected)

    with pytest.raises(KeyError):
        extract_feature_values(features, names, "missing_feature")


def test_compute_ks_test_with_uncertainty():
    rng = np.random.default_rng(42)
    sample_ref = rng.normal(0.0, 1.0, size=200)
    sample_eval = rng.normal(0.0, 1.0, size=200)

    d_stat, d_unc, p_val = compute_ks_test_with_uncertainty(
        sample_ref, sample_eval, n_bootstraps=20, random_seed=42
    )

    assert 0.0 <= d_stat <= 1.0
    assert d_unc >= 0.0
    assert 0.0 <= p_val <= 1.0


def test_compute_wasserstein_with_uncertainty():
    rng = np.random.default_rng(42)
    sample_ref = rng.normal(0.0, 1.0, size=200)
    sample_eval = rng.normal(1.0, 1.0, size=200)

    w_dist, w_unc = compute_wasserstein_with_uncertainty(
        sample_ref, sample_eval, n_bootstraps=20, random_seed=42
    )

    assert w_dist > 0.0
    assert w_unc >= 0.0

