from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from src.utils import (
    compute_js_divergence_with_uncertainty,
    compute_kl_divergence_with_uncertainty,
    compute_ks_test_with_uncertainty,
    compute_wasserstein_with_uncertainty,
    extract_feature_values,
)


class DistributionComparator:
    """Evaluates statistical distance metrics (KS, Wasserstein, KL, and JS divergences) across selections."""

    def __init__(
        self,
        n_bootstraps: int = 100,
        random_seed: int = 42,
        compute_ks: bool = True,
        compute_wasserstein: bool = True,
        compute_kl: bool = True,
        compute_js: bool = True,
    ) -> None:
        """Initializes the comparator with bootstrap parameters and metric flags.

        Args:
            n_bootstraps: Number of bootstrap resamples for uncertainty estimation.
            random_seed: Seed for pseudo-random number generator.
            compute_ks: Whether to compute Kolmogorov-Smirnov distance with uncertainty.
            compute_wasserstein: Whether to compute Wasserstein distance with uncertainty.
            compute_kl: Whether to compute Kullback-Leibler divergence with uncertainty.
            compute_js: Whether to compute Jensen-Shannon divergence with uncertainty.
        """
        self.n_bootstraps = n_bootstraps
        self.random_seed = random_seed
        self.compute_ks = compute_ks
        self.compute_wasserstein = compute_wasserstein
        self.compute_kl = compute_kl
        self.compute_js = compute_js

    def compare_samples(
        self,
        sample_ref: np.ndarray,
        sample_eval: np.ndarray,
        bins: int = 35,
    ) -> Dict[str, Any]:
        """Calculates KS test, Wasserstein distance, KL divergence, and JS divergence.

        Args:
            sample_ref: Reference distribution sample.
            sample_eval: Evaluated model selection sample.
            bins: Bin count for discrete divergences.

        Returns:
            Dictionary containing metrics, uncertainties, and p-value.
        """
        result: Dict[str, Any] = {}
        if self.compute_ks:
            ks_stat, ks_unc, p_val = compute_ks_test_with_uncertainty(
                sample_ref,
                sample_eval,
                n_bootstraps=self.n_bootstraps,
                random_seed=self.random_seed,
            )
            result["ks_stat"] = ks_stat
            result["ks_unc"] = ks_unc
            result["p_value"] = p_val

        if self.compute_wasserstein:
            w_dist, w_unc = compute_wasserstein_with_uncertainty(
                sample_ref,
                sample_eval,
                n_bootstraps=self.n_bootstraps,
                random_seed=self.random_seed,
            )
            result["wasserstein_dist"] = w_dist
            result["wasserstein_unc"] = w_unc

        if self.compute_kl:
            kl_div, kl_unc = compute_kl_divergence_with_uncertainty(
                sample_ref,
                sample_eval,
                bins=bins,
                n_bootstraps=self.n_bootstraps,
                random_seed=self.random_seed,
            )
            result["kl_div"] = kl_div
            result["kl_unc"] = kl_unc

        if self.compute_js:
            js_div, js_unc = compute_js_divergence_with_uncertainty(
                sample_ref,
                sample_eval,
                bins=bins,
                n_bootstraps=self.n_bootstraps,
                random_seed=self.random_seed,
            )
            result["js_div"] = js_div
            result["js_unc"] = js_unc

        return result

    def compare_strategies(
        self,
        features: np.ndarray,
        feature_names: np.ndarray,
        labels: np.ndarray,
        target_features: List[str],
        strategies: Dict[str, np.ndarray],
        reference_strategy: Optional[str] = None,
        is_signal: bool = True,
        bins: int = 35,
    ) -> pd.DataFrame:
        """Compares strategy selections against a ground truth or strategy reference.

        Args:
            features: 2D feature matrix (N, D).
            feature_names: Array of feature names.
            labels: Ground truth binary labels (0 or 1).
            target_features: List of feature names to evaluate.
            strategies: Dict mapping strategy names to binary predictions (N,).
            reference_strategy: Name of reference strategy (None defaults to True Data).
            is_signal: True for signal (class 1), False for background (class 0).
            bins: Bin count for discrete divergences.

        Returns:
            DataFrame of formatted statistical distances with uncertainties.
        """
        class_target = 1 if is_signal else 0
        class_mask = labels == class_target
        records = []

        ref_label = reference_strategy if reference_strategy is not None else ("True Signal" if is_signal else "True Background")

        for feat_name in target_features:
            values = extract_feature_values(features, feature_names, feat_name)
            class_vals = values[class_mask]

            if reference_strategy is None:
                sample_ref = class_vals[np.isfinite(class_vals)]
            else:
                ref_preds = strategies[reference_strategy]
                sample_ref = class_vals[(ref_preds[class_mask] == 1) & np.isfinite(class_vals)]

            for strat_name, preds in strategies.items():
                if strat_name == reference_strategy:
                    continue

                sample_eval = class_vals[(preds[class_mask] == 1) & np.isfinite(class_vals)]
                metrics = self.compare_samples(sample_ref, sample_eval, bins=bins)

                record: Dict[str, Any] = {
                    "Feature": feat_name,
                    "Reference": ref_label,
                    "Strategy": strat_name,
                    "Comparison": f"{strat_name} vs {ref_label}",
                    "Ref Count": len(sample_ref),
                    "Selected Count": len(sample_eval),
                }

                if self.compute_ks:
                    record["KS Distance (D)"] = metrics.get("ks_stat", np.nan)
                    record["KS Uncertainty"] = metrics.get("ks_unc", np.nan)

                if self.compute_wasserstein:
                    record["Wasserstein Distance"] = metrics.get("wasserstein_dist", np.nan)
                    record["Wasserstein Uncertainty"] = metrics.get("wasserstein_unc", np.nan)

                if self.compute_kl:
                    record["KL Divergence"] = metrics.get("kl_div", np.nan)
                    record["KL Uncertainty"] = metrics.get("kl_unc", np.nan)

                if self.compute_js:
                    record["JS Divergence"] = metrics.get("js_div", np.nan)
                    record["JS Uncertainty"] = metrics.get("js_unc", np.nan)

                if self.compute_ks and "p_value" in metrics:
                    record["p-value"] = metrics["p_value"]
                    record["Distribution Equality"] = (
                        "Consistent (p ≥ 0.05)" if metrics["p_value"] >= 0.05 else "Distinct (p < 0.05)"
                    )

                records.append(record)

        return pd.DataFrame(records) if records else pd.DataFrame()
