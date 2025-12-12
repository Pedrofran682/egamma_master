import pandas as pd
import argparse
import os
from src.core.plots import plot_metrics_grid
from src.utils import extract_et, extract_eta, extract_model_name, extract_percentage, get_et_eta, get_results
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


parser = argparse.ArgumentParser(prog='run_roc_plots')
parser.add_argument('results_path',  type=str, required=True)


def run_ringer_plots(results_path: str):
    all_metrics = []
    folders_to_use = [
        "../results/modelV5.dim0.5.folds10_id20251126183757",
        "../results/modelV4.dim0.5.folds10_id20251125151554",
        "../results/modelV3.dim0.5.folds10_id20251117001936",
        "../results/modelV2.dim0.5.folds10_id20251120131323",
    ]
    for folder_path in folders_to_use:
        current_results = get_results(folder_path)
        all_metrics = all_metrics + current_results

    df = pd.DataFrame(all_metrics)
    df['percentage'] = df.apply(extract_percentage, axis=1)
    df['model_name'] = df.apply(extract_model_name, axis=1)
    df['et'] = df.apply(extract_et, axis=1)
    df['eta'] = df.apply(extract_eta, axis=1)

    for model_name in df['model_name'].unique():
        filtered_df = df[df["model_name"] == model_name]
        plot_metrics_grid(filtered_df, model_name, results_path)


if __name__ == "__main__":
    args = parser.parse_args()
    results_path = str(args.results_path)
    run_ringer_plots(results_path)
