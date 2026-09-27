import argparse
import logging
import os
import sys
from datetime import datetime
from logging.config import fileConfig
from pathlib import Path
from typing import List, Optional

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.Plotting.Context import RegionPlotContext
from src.core.Plotting.MetricPlotter import MetricsGridPlotter
from src.utils import (
    extract_et,
    extract_eta,
    extract_model_name,
    extract_percentage,
    get_results,
)

CONFIG_FILE = "logging.ini"
LOG_DIR = "log"
os.makedirs(LOG_DIR, exist_ok=True)
log_filename = f"{LOG_DIR}/plotter_{datetime.now().strftime('%Y%m%d%H%M%S')}.log"

fileConfig(
    CONFIG_FILE,
    defaults={"log_file_path": log_filename},
    disable_existing_loggers=False,
)
log = logging.getLogger(__name__)

parser = argparse.ArgumentParser(prog="run_roc_plots")
parser.add_argument("--results_path", type=str, required=True)
parser.add_argument(
    "--output_dir",
    type=str,
    default=None,
    required=False,
    help="Target directory for plots (defaults to Plots/<target_name>).",
)


class RocPlotRunner:
    """Runner for collecting multi-model training metrics and plotting summary grid comparisons."""

    def __init__(
        self,
        results_path: str,
        folders: Optional[List[str]] = None,
        output_dir: Optional[str] = None,
    ) -> None:
        """Initialize the RocPlotRunner instance.

        Args:
            results_path: Base directory where result files are located.
            folders: List of specific result folders to scan and aggregate.
            output_dir: Target directory for plots (defaults to Plots/<target_name>).
        """
        self.results_path: Path = Path(results_path)
        self.folders: List[str] = folders or [
            "results/modelV5.dim0.5.folds10_id20251126183757",
            "results/modelV4.dim0.5.folds10_id20251125151554",
            "results/modelV3.dim0.5.folds10_id20251117001936",
            "results/modelV2.dim0.5.folds10_id20251120131323",
        ]
        self.plotter: MetricsGridPlotter = MetricsGridPlotter()

        target_name = self.results_path.name or "ROC_Grid"
        if target_name == "results":
            target_name = "ROC_Grid"

        self.output_dir: Path = (
            Path(output_dir) if output_dir else Path("Plots") / target_name
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(self) -> None:
        """Aggregate metric results across folders and generate per-model summary plots."""
        all_metrics = []
        for folder_path in self.folders:
            current_results = get_results(folder_path)
            all_metrics.extend(current_results)

        df = pd.DataFrame(all_metrics)
        df["percentage"] = df.apply(extract_percentage, axis=1)
        df["model_name"] = df.apply(extract_model_name, axis=1)
        df["et"] = df.apply(extract_et, axis=1)
        df["eta"] = df.apply(extract_eta, axis=1)

        dummy_context = RegionPlotContext(
            iet=0,
            ieta=0,
            output_dir=str(self.output_dir),
            data=None,
            target=None,
        )

        for model_name in df["model_name"].unique():
            filtered_df = df[df["model_name"] == model_name]
            self.plotter.plot(dummy_context, df=filtered_df, model_name=model_name)


if __name__ == "__main__":
    args = parser.parse_args()
    runner = RocPlotRunner(
        results_path=args.results_path,
        output_dir=args.output_dir,
    )
    runner.run()
