import importlib
import logging
import os
import re
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch

from src.Models.Models import get_model
from src.Parser.DynamicConfiguration import DynamicConfiguration

log = logging.getLogger()


def get_best_sp_model(
    training_results_list: pd.DataFrame, model_builder_tag: str, input_dimensions: int
):
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
    folder_path = Path(base_path) / new_folder_name
    try:
        folder_path.mkdir(parents=True, exist_ok=True)
        log.info(f"Folder '{folder_path}' created or already exists (using pathlib).")
    except OSError as error:
        log.error(f"Error creating directory '{folder_path}': {error}")
    return str(folder_path)


def norm1(data: np.ndarray) -> np.ndarray:
    norms = np.abs(data.sum(axis=1))
    norms[norms == 0] = 1
    return data / norms[:, None]


def compute_saliency_map(x: np.ndarray, model: torch.nn.Module) -> np.ndarray:
    x_tensor = torch.from_numpy(x[np.newaxis, :]).float().requires_grad_(True)
    pred = model(x_tensor)
    grad = torch.autograd.grad(outputs=pred.sum(), inputs=x_tensor)[0]
    saliency = np.abs(grad.numpy()[0])
    smin, smax = saliency.min(), saliency.max()
    saliency_norm = (saliency - smin) / (smax - smin + 1e-8)
    return saliency_norm


def compute_mean_std_saliency(
    X_subset: np.ndarray, model: torch.nn.Module
) -> Tuple[float, float]:
    saliency_list = [compute_saliency_map(x, model) for x in X_subset]
    saliency_array = np.stack(saliency_list)
    return np.mean(saliency_array, axis=0), np.std(saliency_array, axis=0)


def get_results_file_name(et: int, eta: int, repeat: int, fold_idx: int) -> str:
    return "repeat{repeat}.fold_idx{fold_idx}.iet{iet}.ieta{ieta}.pkl".format(
        ieta=eta, iet=et, repeat=repeat, fold_idx=fold_idx
    )


def verify_results(
    folder_path: str, et: int, eta: int, repeat: int, fold_idx: int
) -> bool:
    file2verify = os.path.join(
        folder_path, get_results_file_name(et, eta, repeat, fold_idx)
    )
    log.info(f"Verifying {file2verify}")
    if os.path.exists(file2verify):
        log.info(f"{file2verify} already processed")
        return True
    return False


def get_et_eta(file_path) -> Tuple[int, int]:
    regex_pattern = r"et(\d+).*?eta(\d+)"
    match = re.search(regex_pattern, file_path)

    if match:
        et = match.group(1)
        eta = match.group(2)
        return int(et), int(eta)
    raise Exception("Could get et or eta value from file path.")


def get_results(folder_path: str):
    listed_files = [
        file
        for file in os.listdir(folder_path)
        if (file.endswith(".pkl") and not file.startswith(".sys.v#."))
    ]
    all_metrics = []
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
        metrics = {
            "source_file": None,
            "mean_sp": 0,
            "std_sp": 0,
            "mean_fa": 0,
            "std_fa": 0,
            "mean_pd": 0,
            "std_pd": 0,
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


def get_metrics(df: pd.DataFrame, column: str):
    mean_value = df[column].mean()
    std_value = df[column].std()
    return mean_value, std_value


def extract_percentage(row) -> str:
    match = re.search(r"dim(\d+\.\d+)", row["folder_path"])
    if match:
        return match.group(1)
    return "extract_percentage function failed"


def extract_model_name(row) -> str:
    match = re.search(r"(modelV\d+)", row["folder_path"])
    if match:
        return match.group(1)
    return "extract_model_name function failed"


def extract_et(row) -> str:
    match = re.search(r"iet(\d+)\.ieta(\d+)", row["source_file"])
    if match:
        return match.group(1)
    return "extract_et function failed"


def extract_eta(row) -> str:
    match = re.search(r"iet(\d+)\.ieta(\d+)", row["source_file"])
    if match:
        return match.group(2)
    return "extract_eta function failed"


def get_instance(configuration: DynamicConfiguration):
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
