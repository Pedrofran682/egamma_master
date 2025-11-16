import pandas as pd
import argparse
import numpy as np
import os
from src.core.plots import plot_model_metrics, plot_model_acc, plot_profile_mean_energy_rings
from src.utils import get_et_eta
from src.core.EgammaNpzDataset import EgammaNpzDataset
from datetime import datetime
import logging
from logging.config import fileConfig


CONFIG_FILE = 'logging.ini'
LOG_DIR = 'log'
os.makedirs(LOG_DIR, exist_ok=True)
log_filename = f"{LOG_DIR}/plotter_{datetime.now().strftime('%Y%m%d%H%M%S')}.log"

fileConfig(CONFIG_FILE, 
           defaults={'log_file_path': log_filename})
log = logging.getLogger(__name__)


parser = argparse.ArgumentParser(prog='plotter')
parser.add_argument('results_path',  type=str)
parser.add_argument('--drive_path',  default="data/", type=str,required=False)
parser.add_argument('--plot_ringer', default=False, type=bool, required=False)
parser.add_argument('--debug', action='store_true')


def run_ringer_plots(drive_path: str, folder_path: str,
                     et: int, eta: int) -> None:
    data_folder = [os.path.join(drive_path, file) for file in os.listdir(drive_path)
                    if (file.endswith(".npz") and 
                        file.startswith("mc23_13TeV"))
                        ]
    egammaDataset = EgammaNpzDataset(data_folder, 
                                    percentage=1.0)
    data, target, file_name = egammaDataset[0]
    et, eta = get_et_eta(file_name)
    plot_profile_mean_energy_rings(data, target, egammaDataset.indexes,
                                folder_path=folder_path, iet=et, ieta=eta)
    

def run_plots(results_data: str, folder_path: str,
              et: int, eta: int) -> None:
    data = pd.read_pickle(results_data)
    data = pd.DataFrame(data)
    data = data.iloc[np.argmax(data["best_sp_value"])]

    plot_model_metrics(data["history"], folder_path, et, eta)
    plot_model_acc(data["history"]["callbackMetrics"], folder_path, et, eta)


if __name__ == "__main__":
    args = parser.parse_args()
    results_path = str(args.results_path)
    drive_path = str(args.drive_path)
    data_folder = [os.path.join(results_path, file) for file in os.listdir(results_path)
                if (file.endswith(".pkl") and file.startswith("ie"))
                ]
    for file_path in data_folder:
        log.info(f"Plotando resultado de {file_path}")
        folder_path = os.path.dirname(file_path)
        et, eta = get_et_eta(file_path)
        run_plots(file_path, folder_path, et, eta)
    if args.plot_ringer:
        log.info(f"Executando plots do perfil dos anéis")
        run_ringer_plots(drive_path, folder_path, et, eta)