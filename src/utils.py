from pathlib import Path
import numpy as np
from typing import  Tuple
import torch
import logging
import re
import os
from src.Architectures.Models import get_model
import pandas as pd
log = logging.getLogger()


def get_best_sp_model(training_results_list: pd.DataFrame,
                      model_builder_tag: str,
                      input_dimensions: int) :
    if not (training_results_list.shape[0] > 0):
        log.error("A lista de resultados de treinamento está vazia.")
        return None, None

    highest_sp_model_info = training_results_list.iloc[np.argmax(training_results_list["best_sp_value"])]

    try:
        best_overall_sp_model = get_model(model_builder_tag, input_dimensions)
        best_overall_sp_model.load_state_dict(highest_sp_model_info["best_weights"])
        return best_overall_sp_model, highest_sp_model_info
    except Exception as e:
        log.info(f"Não foi possível carregar o modelo com os pesos fornecidos. Motivo: {e}")
        raise e

def create_folder(new_folder_name: str , base_path: str = "results") -> str:
    folder_path = Path(base_path) / new_folder_name 
    try:
        folder_path.mkdir(parents=True, exist_ok=True)
        log.info(f"Folder '{folder_path}' created or already exists (using pathlib).")
    except OSError as error:
        log.error(f"Error creating directory '{folder_path}': {error}")
    return str(folder_path)


def norm1(data: np.ndarray) -> np.ndarray:
      norms = np.abs( data.sum(axis=1) )
      norms[norms==0] = 1
      return data / norms[:, None]

def compute_saliency_map(x: np.ndarray, model: torch.nn.Module) -> np.ndarray:
    x_tensor = torch.from_numpy(x[np.newaxis, :]).float().requires_grad_(True)
    pred = model(x_tensor)
    grad = torch.autograd.grad(outputs=pred.sum(), inputs=x_tensor)[0]
    saliency = np.abs(grad.numpy()[0])
    smin, smax = saliency.min(), saliency.max()
    saliency_norm = (saliency - smin) / (smax - smin + 1e-8)
    return saliency_norm


def compute_mean_std_saliency(X_subset: np.ndarray, 
                              model: torch.nn.Module) -> Tuple[float, float]:
    saliency_list = [compute_saliency_map(x, model) for x in X_subset]
    saliency_array = np.stack(saliency_list)
    return np.mean(saliency_array, axis=0), np.std(saliency_array, axis=0)


def get_results_file_name(et: int, eta: int) -> str:
        return "iet{iet}.ieta{ieta}.pkl".format(ieta = eta,
                                                iet = et)
    
def verify_results(folder_path: str, et: int, eta: int) -> bool:
    file2verify = os.path.join(folder_path, get_results_file_name(et, eta))
    log.info(f"Verifying {file2verify}")
    if os.path.exists(file2verify):
        log.info(f"{file2verify} already processed")
        return False
    return True

def get_et_eta(file_path):
    regex_pattern = r"et(\d+).*?eta(\d+)"
    match = re.search(regex_pattern, file_path)
    
    if match:
        et = match.group(1)
        eta = match.group(2)
        return et, eta
