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

    def __init__(self, n_bootstraps: int = 100, random_seed: int = 42) -> None:
        """Initializes the comparator with bootstrap parameters.

        Args:
            n_bootstraps: Number of bootstrap resamples for uncertainty estimation.
            random_seed: Seed for pseudo-random number generator.
        """
        self.n_bootstraps = n_bootstraps
        self.random_seed = random_seed

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
        ks_stat, ks_unc, p_val = compute_ks_test_with_uncertainty(
            sample_ref,
            sample_eval,
            n_bootstraps=self.n_bootstraps,
            random_seed=self.random_seed,
        )
        w_dist, w_unc = compute_wasserstein_with_uncertainty(
            sample_ref,
            sample_eval,
            n_bootstraps=self.n_bootstraps,
            random_seed=self.random_seed,
        )
        kl_div, kl_unc = compute_kl_divergence_with_uncertainty(
            sample_ref,
            sample_eval,
            bins=bins,
            n_bootstraps=self.n_bootstraps,
            random_seed=self.random_seed,
        )
        js_div, js_unc = compute_js_divergence_with_uncertainty(
            sample_ref,
            sample_eval,
            bins=bins,
            n_bootstraps=self.n_bootstraps,
            random_seed=self.random_seed,
        )

        return {
            "ks_stat": ks_stat,
            "ks_unc": ks_unc,
            "p_value": p_val,
            "wasserstein_dist": w_dist,
            "wasserstein_unc": w_unc,
            "kl_div": kl_div,
            "kl_unc": kl_unc,
            "js_div": js_div,
            "js_unc": js_unc,
        }

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

                records.append({
                    "Feature": feat_name,
                    "Reference": ref_label,
                    "Strategy": strat_name,
                    "Comparison": f"{strat_name} vs {ref_label}",
                    "Ref Count": len(sample_ref),
                    "Selected Count": len(sample_eval),
                    "KS Distance (D)": f"{metrics['ks_stat']:.4f} ± {metrics['ks_unc']:.4f}",
                    "Wasserstein Distance": f"{metrics['wasserstein_dist']:.4g} ± {metrics['wasserstein_unc']:.4g}",
                    "KL Divergence": f"{metrics['kl_div']:.4g} ± {metrics['kl_unc']:.4g}",
                    "JS Divergence": f"{metrics['js_div']:.4g} ± {metrics['js_unc']:.4g}",
                    "p-value": f"{metrics['p_value']:.3e}",
                    "Distribution Equality": "Consistent (p ≥ 0.05)" if metrics["p_value"] >= 0.05 else "Distinct (p < 0.05)",
                })

        return pd.DataFrame(records) if records else pd.DataFrame()
