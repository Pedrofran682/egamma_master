import importlib
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import special, stats
import torch
from torch.utils.data import WeightedRandomSampler

from src.Models.Models import get_model
from src.Parser.DynamicConfiguration import DynamicConfiguration

log = logging.getLogger()


def get_best_sp_model(
    training_results_list: pd.DataFrame,
    model_builder_tag: str,
    input_dimensions: int,
) -> Tuple[Optional[torch.nn.Module], Optional[pd.Series]]:
    """Retrieve the model and checkpoint info with the highest SP value.

    Args:
        training_results_list: DataFrame containing training fold records.
        model_builder_tag: Architecture identifier tag.
        input_dimensions: Number of input features for the network.

    Returns:
        Tuple containing the instantiated model with loaded weights and the record series.
    """
    if not (training_results_list.shape[0] > 0):
        log.error("A lista de resultados de treinamento está vazia.")
        return None, None

    highest_sp_model_info = training_results_list.iloc[
        np.argmax(training_results_list["best_sp_value"])
    ]

    try:
        best_overall_sp_model = get_model(model_builder_tag, input_dimensions)
        best_overall_sp_model.load_state_dict(highest_sp_model_info["best_weights"])
        return best_overall_sp_model, highest_sp_model_info
    except Exception as e:
        log.info(
            f"Não foi possível carregar o modelo com os pesos fornecidos. Motivo: {e}"
        )
        raise e


def create_folder(new_folder_name: str, base_path: str = "results") -> str:
    """Create a directory if it does not exist.

    Args:
        new_folder_name: Subdirectory name to create.
        base_path: Root parent path.

    Returns:
        Created directory path as a string.
    """
    folder_path = Path(base_path) / new_folder_name
    try:
        folder_path.mkdir(parents=True, exist_ok=True)
        log.info(f"Folder '{folder_path}' created or already exists.")
    except OSError as error:
        log.error(f"Error creating directory '{folder_path}': {error}")
    return str(folder_path)


def norm1(data: np.ndarray) -> np.ndarray:
    """Normalize input samples along axis 1 by their L1 norm.

    Args:
        data: 2D array of feature vectors.

    Returns:
        L1-normalized array where row sums equal 1.
    """
    norms = np.abs(data.sum(axis=1))
    norms[norms == 0] = 1
    return data / norms[:, None]


def compute_saliency_map(x: np.ndarray, model: torch.nn.Module) -> np.ndarray:
    """Compute normalized gradient saliency map for a single sample.

    Args:
        x: 1D feature array for a single event.
        model: Trained PyTorch neural network.

    Returns:
        Min-max normalized 1D saliency array across input features.
    """
    x_tensor = torch.from_numpy(x[np.newaxis, :]).float().requires_grad_(True)
    pred = model(x_tensor)
    grad = torch.autograd.grad(outputs=pred.sum(), inputs=x_tensor)[0]
    saliency = np.abs(grad.numpy()[0])
    smin, smax = saliency.min(), saliency.max()
    saliency_norm = (saliency - smin) / (smax - smin + 1e-8)
    return saliency_norm


def compute_mean_std_saliency(
    X_subset: np.ndarray,
    model: torch.nn.Module,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute mean and standard deviation saliency profiles across a sample subset.

    Args:
        X_subset: 2D array of samples.
        model: PyTorch neural network.

    Returns:
        Tuple of (mean_saliency, std_saliency) 1D arrays.
    """
    saliency_list = [compute_saliency_map(x, model) for x in X_subset]
    saliency_array = np.stack(saliency_list)
    return np.mean(saliency_array, axis=0), np.std(saliency_array, axis=0)


def get_results_file_name(et: int, eta: int, repeat: int, fold_idx: int) -> str:
    """Format standard pickle filename for a fold result.

    Args:
        et: Transverse energy bin index.
        eta: Pseudorapidity bin index.
        repeat: Repetition run index.
        fold_idx: Cross-validation fold index.

    Returns:
        Formatted filename string.
    """
    return f"repeat{repeat}.fold_idx{fold_idx}.iet{et}.ieta{eta}.pkl"


def verify_results(
    folder_path: str, et: int, eta: int, repeat: int, fold_idx: int
) -> bool:
    """Check if a specific fold result pickle file exists.

    Args:
        folder_path: Destination results folder.
        et: Transverse energy bin index.
        eta: Pseudorapidity bin index.
        repeat: Repetition run index.
        fold_idx: Cross-validation fold index.

    Returns:
        True if the result file exists, False otherwise.
    """
    file2verify = os.path.join(
        folder_path, get_results_file_name(et, eta, repeat, fold_idx)
    )
    log.info(f"Verifying {file2verify}")
    if os.path.exists(file2verify):
        log.info(f"{file2verify} already processed")
        return True
    return False


def get_et_eta(file_path: str) -> Tuple[int, int]:
    """Extract et and eta bin indices from a file path string.

    Args:
        file_path: File name or path containing 'et' and 'eta' tokens.

    Returns:
        Tuple of integer (et, eta) indices.

    Raises:
        ValueError: If et or eta cannot be parsed from the path.
    """
    regex_pattern = r"et(\d+).*?eta(\d+)"
    match = re.search(regex_pattern, file_path)

    if match:
        et = match.group(1)
        eta = match.group(2)
        return int(et), int(eta)
    raise ValueError(f"Could not get et or eta value from file path: {file_path}")


def get_results(folder_path: str) -> List[Dict[str, Any]]:
    """Parse and summarize all fold results from a results directory.

    Args:
        folder_path: Path containing pickle result files.

    Returns:
        List of dictionaries with aggregated metric summaries for each file.
    """
    listed_files = [
        file
        for file in os.listdir(folder_path)
        if (file.endswith(".pkl") and not file.startswith(".sys.v#."))
    ]
    all_metrics: List[Dict[str, Any]] = []
    for file in listed_files:
        data = pd.DataFrame(pd.read_pickle(os.path.join(folder_path, file)))
        data = data[
            [
                "file_path",
                "fold",
                "best_sp_value",
                "best_fa_value",
                "best_pd_value",
                "history",
            ]
        ]
        metrics: Dict[str, Any] = {
            "source_file": None,
            "mean_sp": 0.0,
            "std_sp": 0.0,
            "mean_fa": 0.0,
            "std_fa": 0.0,
            "mean_pd": 0.0,
            "std_pd": 0.0,
        }

        metrics["mean_sp"], metrics["std_sp"] = get_metrics(data, "best_sp_value")
        metrics["mean_fa"], metrics["std_fa"] = get_metrics(data, "best_fa_value")
        metrics["mean_pd"], metrics["std_pd"] = get_metrics(data, "best_pd_value")
        metrics["source_file"] = file

        metrics["folder_path"] = folder_path
        max_score_index = data["best_sp_value"].idxmax()
        row_of_max_value = data.loc[max_score_index]
        metrics["auc"] = row_of_max_value["history"]["callbackMetrics"]["auc_score"]
        metrics["best_sp_value"] = row_of_max_value["best_sp_value"]
        metrics["best_fa_value"] = row_of_max_value["best_fa_value"]
        metrics["best_pd_value"] = row_of_max_value["best_pd_value"]

        all_metrics.append(metrics)
    return all_metrics


def get_metrics(df: pd.DataFrame, column: str) -> Tuple[float, float]:
    """Calculate mean and standard deviation for a DataFrame column.

    Args:
        df: Input DataFrame.
        column: Target column name.

    Returns:
        Tuple of (mean, standard_deviation).
    """
    mean_value = float(df[column].mean())
    std_value = float(df[column].std())
    return mean_value, std_value


def extract_percentage(row: pd.Series) -> str:
    """Extract ring percentage identifier from folder_path in a series row.

    Args:
        row: Series containing a 'folder_path' entry.

    Returns:
        Extracted percentage string, or fallback failure message.
    """
    match = re.search(r"dim(\d+\.\d+)", row["folder_path"])
    if match:
        return match.group(1)
    return "extract_percentage function failed"


def extract_model_name(row: pd.Series) -> str:
    """Extract model name identifier from folder_path in a series row.

    Args:
        row: Series containing a 'folder_path' entry.

    Returns:
        Extracted model name string, or fallback failure message.
    """
    match = re.search(r"(modelV\d+)", row["folder_path"])
    if match:
        return match.group(1)
    return "extract_model_name function failed"


def extract_et(row: pd.Series) -> str:
    """Extract et index from source_file name in a series row.

    Args:
        row: Series containing a 'source_file' entry.

    Returns:
        Extracted et index string, or fallback failure message.
    """
    match = re.search(r"iet(\d+)\.ieta(\d+)", row["source_file"])
    if match:
        return match.group(1)
    return "extract_et function failed"


def extract_eta(row: pd.Series) -> str:
    """Extract eta index from source_file name in a series row.

    Args:
        row: Series containing a 'source_file' entry.

    Returns:
        Extracted eta index string, or fallback failure message.
    """
    match = re.search(r"iet(\d+)\.ieta(\d+)", row["source_file"])
    if match:
        return match.group(2)
    return "extract_eta function failed"


def get_instance(configuration: DynamicConfiguration) -> Any:
    """Instantiate a class dynamically from configuration parameters.

    Args:
        configuration: DynamicConfiguration specifying module, object, and parameters.

    Returns:
        Instantiated class object.

    Raises:
        ValueError: If module cannot be imported or object is missing.
    """
    try:
        module = importlib.import_module(configuration.module)
        target_class = getattr(module, configuration.object_name)
        return target_class(**configuration.parameters)
    except ImportError:
        raise ValueError(f"Could not import module '{configuration.module}'")
    except AttributeError:
        raise ValueError(
            f"Could not find '{configuration.object_name}' in '{configuration.module}'"
        )
    except Exception as e:
        raise ValueError(f"Error initializing {configuration.object_name}: {e}")


def get_class_weight(target: Union[np.ndarray, torch.Tensor]) -> WeightedRandomSampler:
    """Create a WeightedRandomSampler to balance class frequencies.

    Args:
        target: Target label array or tensor (binary/multiclass).

    Returns:
        Configured PyTorch WeightedRandomSampler.
    """
    log.info("Creating samples_weight")
    if isinstance(target, torch.Tensor):
        target_np = target.detach().cpu().numpy().flatten()
    else:
        target_np = target.flatten()

    class_sample_count = np.array(
        [len(np.where(target_np == t)[0]) for t in np.unique(target_np)]
    )
    weight = 1.0 / class_sample_count
    samples_weight = np.array([weight[int(t)] for t in target_np])

    samples_weight_tensor = torch.from_numpy(samples_weight)
    number_generator = torch.Generator().manual_seed(42)
    return WeightedRandomSampler(
        samples_weight_tensor,  # type: ignore
        len(samples_weight_tensor),
        generator=number_generator,
    )


def save_table(
    table: pd.DataFrame,
    directory: Union[str, Path],
    region_key: str,
    name: str,
) -> Path:
    """Persists a results table as CSV with a region-prefixed filename.

    Args:
        table: DataFrame to persist.
        directory: Destination folder (created if missing).
        region_key: Kinematic region identifier used as filename prefix.
        name: Descriptive table name.

    Returns:
        Path of the written CSV file.
    """
    dir_path = Path(directory)
    dir_path.mkdir(parents=True, exist_ok=True)
    csv_path = dir_path / f"{region_key}_{name}.csv"
    table.to_csv(csv_path, index=False)
    log.info(f"Saved table: {csv_path}")
    return csv_path


def extract_feature_values(
    features: np.ndarray,
    names: np.ndarray,
    feat_name: str,
) -> np.ndarray:
    """Extracts raw feature column or calculates derived physics variables like Eratio.

    Args:
        features: Event feature matrix of shape (N, D).
        names: Column feature names array of shape (D,).
        feat_name: Target variable name.

    Returns:
        1D numpy array of values.
    """
    name_to_idx = {name.strip(): i for i, name in enumerate(names)}

    if feat_name.lower() in ["eratio", "eration", "l2_eratio"]:
        if "trig_L2_calo_emaxs1" in name_to_idx and "trig_L2_calo_e2tsts1" in name_to_idx:
            emaxs1 = features[:, name_to_idx["trig_L2_calo_emaxs1"]]
            e2tsts1 = features[:, name_to_idx["trig_L2_calo_e2tsts1"]]
            denom = emaxs1 + e2tsts1
            safe_denom = np.where(denom != 0.0, denom, 1.0)
            return np.where(denom != 0.0, (emaxs1 - e2tsts1) / safe_denom, 0.0)
        raise KeyError("Cannot compute Eratio: 'trig_L2_calo_emaxs1' and 'trig_L2_calo_e2tsts1' not found.")

    if feat_name in name_to_idx:
        return features[:, name_to_idx[feat_name]]

    for name, idx in name_to_idx.items():
        if feat_name.lower() == name.lower():
            return features[:, idx]

    raise KeyError(f"Feature '{feat_name}' not found in dataset columns.")


def compute_ks_test_with_uncertainty(
    sample_ref: np.ndarray,
    sample_eval: np.ndarray,
    n_bootstraps: int = 100,
    random_seed: int = 42,
) -> Tuple[float, float, float]:
    """Computes two-sample Kolmogorov-Smirnov statistic D, bootstrap uncertainty, and p-value.

    Args:
        sample_ref: Reference distribution sample.
        sample_eval: Evaluated model-selected sample.
        n_bootstraps: Number of bootstrap iterations for uncertainty estimation.
        random_seed: Random seed for reproducibility.

    Returns:
        Tuple of (ks_statistic, ks_uncertainty, p_value).
    """
    sample_ref = sample_ref[np.isfinite(sample_ref)]
    sample_eval = sample_eval[np.isfinite(sample_eval)]
    if len(sample_ref) == 0 or len(sample_eval) == 0:
        return 0.0, 0.0, 1.0

    ks_res = stats.ks_2samp(sample_ref, sample_eval)
    d_stat = float(ks_res.statistic)
    p_val = float(ks_res.pvalue)

    rng = np.random.default_rng(random_seed)
    n_ref = len(sample_ref)
    n_eval = len(sample_eval)
    d_boot = np.empty(n_bootstraps, dtype=float)
    for b in range(n_bootstraps):
        b_ref = rng.choice(sample_ref, size=n_ref, replace=True)
        b_eval = rng.choice(sample_eval, size=n_eval, replace=True)
        d_boot[b] = stats.ks_2samp(b_ref, b_eval, method="asymp").statistic
    d_unc = float(np.std(d_boot, ddof=1))
    return d_stat, d_unc, p_val


def compute_wasserstein_with_uncertainty(
    sample_ref: np.ndarray,
    sample_eval: np.ndarray,
    n_bootstraps: int = 100,
    random_seed: int = 42,
) -> Tuple[float, float]:
    """Computes first Wasserstein distance and bootstrap uncertainty.

    Args:
        sample_ref: Reference distribution sample.
        sample_eval: Evaluated model-selected sample.
        n_bootstraps: Number of bootstrap iterations for uncertainty estimation.
        random_seed: Random seed for reproducibility.

    Returns:
        Tuple of (wasserstein_distance, wasserstein_uncertainty).
    """
    sample_ref = sample_ref[np.isfinite(sample_ref)]
    sample_eval = sample_eval[np.isfinite(sample_eval)]
    if len(sample_ref) == 0 or len(sample_eval) == 0:
        return 0.0, 0.0

    w_dist = float(stats.wasserstein_distance(sample_ref, sample_eval))

    rng = np.random.default_rng(random_seed)
    n_ref = len(sample_ref)
    n_eval = len(sample_eval)
    w_boot = np.empty(n_bootstraps, dtype=float)
    for b in range(n_bootstraps):
        b_ref = rng.choice(sample_ref, size=n_ref, replace=True)
        b_eval = rng.choice(sample_eval, size=n_eval, replace=True)
        w_boot[b] = stats.wasserstein_distance(b_ref, b_eval)
    w_unc = float(np.std(w_boot, ddof=1))
    return w_dist, w_unc


def _estimate_empirical_pmf(
    sample_ref: np.ndarray,
    sample_eval: np.ndarray,
    bins: Union[int, np.ndarray] = 35,
    epsilon: float = 1e-7,
) -> Tuple[np.ndarray, np.ndarray]:
    """Discretizes continuous samples onto common support with Laplace/epsilon smoothing."""
    sample_ref = sample_ref[np.isfinite(sample_ref)]
    sample_eval = sample_eval[np.isfinite(sample_eval)]

    if isinstance(bins, int):
        all_vals = np.concatenate([sample_ref, sample_eval])
        low, high = float(np.percentile(all_vals, 0.5)), float(np.percentile(all_vals, 99.5))
        if low >= high:
            low -= 1.0
            high += 1.0
        bin_edges = np.linspace(low, high, bins + 1)
    else:
        bin_edges = np.asarray(bins, dtype=float)

    c_ref, _ = np.histogram(sample_ref, bins=bin_edges)
    c_eval, _ = np.histogram(sample_eval, bins=bin_edges)

    p = (c_ref + epsilon) / (c_ref.sum() + epsilon * len(c_ref))
    q = (c_eval + epsilon) / (c_eval.sum() + epsilon * len(c_eval))
    return p, q


def compute_kl_divergence_with_uncertainty(
    sample_ref: np.ndarray,
    sample_eval: np.ndarray,
    bins: Union[int, np.ndarray] = 35,
    n_bootstraps: int = 100,
    random_seed: int = 42,
    epsilon: float = 1e-7,
) -> Tuple[float, float]:
    """Computes discrete Kullback-Leibler divergence D_KL(P || Q) and bootstrap uncertainty.

    Args:
        sample_ref: Reference distribution sample P.
        sample_eval: Evaluated model selection sample Q.
        bins: Number of bins or explicit bin edges.
        n_bootstraps: Number of bootstrap iterations.
        random_seed: Random seed for reproducibility.
        epsilon: Regularization smoothing constant to avoid log(0).

    Returns:
        Tuple of (kl_divergence, kl_uncertainty).
    """
    sample_ref = sample_ref[np.isfinite(sample_ref)]
    sample_eval = sample_eval[np.isfinite(sample_eval)]
    if len(sample_ref) == 0 or len(sample_eval) == 0:
        return 0.0, 0.0

    p, q = _estimate_empirical_pmf(sample_ref, sample_eval, bins=bins, epsilon=epsilon)
    kl_stat = float(np.sum(special.rel_entr(p, q)))

    if isinstance(bins, int):
        all_vals = np.concatenate([sample_ref, sample_eval])
        low, high = float(np.percentile(all_vals, 0.5)), float(np.percentile(all_vals, 99.5))
        if low >= high:
            low -= 1.0
            high += 1.0
        fixed_bin_edges = np.linspace(low, high, bins + 1)
    else:
        fixed_bin_edges = bins

    rng = np.random.default_rng(random_seed)
    n_ref = len(sample_ref)
    n_eval = len(sample_eval)
    kl_boot = np.empty(n_bootstraps, dtype=float)

    for b in range(n_bootstraps):
        b_ref = rng.choice(sample_ref, size=n_ref, replace=True)
        b_eval = rng.choice(sample_eval, size=n_eval, replace=True)
        bp, bq = _estimate_empirical_pmf(b_ref, b_eval, bins=fixed_bin_edges, epsilon=epsilon)
        kl_boot[b] = float(np.sum(special.rel_entr(bp, bq)))

    kl_unc = float(np.std(kl_boot, ddof=1))
    return kl_stat, kl_unc


def compute_js_divergence_with_uncertainty(
    sample_ref: np.ndarray,
    sample_eval: np.ndarray,
    bins: Union[int, np.ndarray] = 35,
    n_bootstraps: int = 100,
    random_seed: int = 42,
    epsilon: float = 1e-7,
) -> Tuple[float, float]:
    """Computes symmetric Jensen-Shannon divergence in [0, 1] and bootstrap uncertainty.

    Args:
        sample_ref: Reference distribution sample P.
        sample_eval: Evaluated model selection sample Q.
        bins: Number of bins or explicit bin edges.
        n_bootstraps: Number of bootstrap iterations.
        random_seed: Random seed for reproducibility.
        epsilon: Regularization smoothing constant.

    Returns:
        Tuple of (js_divergence, js_uncertainty).
    """
    sample_ref = sample_ref[np.isfinite(sample_ref)]
    sample_eval = sample_eval[np.isfinite(sample_eval)]
    if len(sample_ref) == 0 or len(sample_eval) == 0:
        return 0.0, 0.0

    p, q = _estimate_empirical_pmf(sample_ref, sample_eval, bins=bins, epsilon=epsilon)
    m = 0.5 * (p + q)
    js_stat = float((0.5 * np.sum(special.rel_entr(p, m)) + 0.5 * np.sum(special.rel_entr(q, m))) / np.log(2))

    if isinstance(bins, int):
        all_vals = np.concatenate([sample_ref, sample_eval])
        low, high = float(np.percentile(all_vals, 0.5)), float(np.percentile(all_vals, 99.5))
        if low >= high:
            low -= 1.0
            high += 1.0
        fixed_bin_edges = np.linspace(low, high, bins + 1)
    else:
        fixed_bin_edges = bins

    rng = np.random.default_rng(random_seed)
    n_ref = len(sample_ref)
    n_eval = len(sample_eval)
    js_boot = np.empty(n_bootstraps, dtype=float)

    for b in range(n_bootstraps):
        b_ref = rng.choice(sample_ref, size=n_ref, replace=True)
        b_eval = rng.choice(sample_eval, size=n_eval, replace=True)
        bp, bq = _estimate_empirical_pmf(b_ref, b_eval, bins=fixed_bin_edges, epsilon=epsilon)
        bm = 0.5 * (bp + bq)
        js_val = (0.5 * np.sum(special.rel_entr(bp, bm)) + 0.5 * np.sum(special.rel_entr(bq, bm))) / np.log(2)
        js_boot[b] = float(js_val)

    js_unc = float(np.std(js_boot, ddof=1))
    return js_stat, js_unc
