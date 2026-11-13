import numpy as np
import pytest

from src.core.Validation.DistributionComparator import DistributionComparator


def test_distribution_comparator_compare_samples():
    comparator = DistributionComparator(n_bootstraps=20, random_seed=42)

    rng = np.random.default_rng(42)
    sample_ref = rng.normal(0.0, 1.0, size=150)
    sample_eval = rng.normal(0.2, 1.0, size=150)

    metrics = comparator.compare_samples(sample_ref, sample_eval, bins=15)

    assert "ks_stat" in metrics
    assert "wasserstein_dist" in metrics
    assert "kl_div" in metrics
    assert "js_div" in metrics
    assert 0.0 <= metrics["ks_stat"] <= 1.0
    assert metrics["wasserstein_dist"] >= 0.0
    assert metrics["kl_div"] >= 0.0
    assert 0.0 <= metrics["js_div"] <= 1.0


def test_distribution_comparator_compare_strategies():
    comparator = DistributionComparator(n_bootstraps=15, random_seed=42)

    rng = np.random.default_rng(42)
    n = 200
    features = np.column_stack([
        rng.normal(0.0, 1.0, size=n),
        rng.uniform(10.0, 50.0, size=n),
    ])
    feature_names = np.array(["feat_a", "feat_b"])
    labels = rng.binomial(1, 0.5, size=n)

    strat_1 = rng.binomial(1, 0.85, size=n)
    strat_2 = rng.binomial(1, 0.70, size=n)

    strategies = {
        "Neural Model": strat_1,
        "Athena Cut": strat_2,
    }

    # 1. Compare vs True Data
    df_true = comparator.compare_strategies(
        features=features,
        feature_names=feature_names,
        labels=labels,
        target_features=["feat_a"],
        strategies=strategies,
        reference_strategy=None,
        is_signal=True,
    )
    assert len(df_true) == 2
    assert "KS Distance (D)" in df_true.columns
    assert "KS Uncertainty" in df_true.columns
    assert "Wasserstein Distance" in df_true.columns
    assert "Wasserstein Uncertainty" in df_true.columns
    assert "KL Divergence" in df_true.columns
    assert "KL Uncertainty" in df_true.columns
    assert "JS Divergence" in df_true.columns
    assert "JS Uncertainty" in df_true.columns
    assert isinstance(df_true.iloc[0]["KS Distance (D)"], float)
    assert isinstance(df_true.iloc[0]["KS Uncertainty"], float)

    # 2. Compare vs Athena Cut
    df_athena = comparator.compare_strategies(
        features=features,
        feature_names=feature_names,
        labels=labels,
        target_features=["feat_a"],
        strategies=strategies,
        reference_strategy="Athena Cut",
        is_signal=True,
    )
    assert len(df_athena) == 1
    assert df_athena.iloc[0]["Strategy"] == "Neural Model"
    assert df_athena.iloc[0]["Reference"] == "Athena Cut"
