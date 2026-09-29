import argparse
import logging
import os
import sys
import re
from datetime import datetime
from logging.config import fileConfig
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.Datasets.EgammaNpzDataset import EgammaNpzDataset
from src.core.Plotting.Context import RegionPlotContext
from src.core.Plotting.PlotManager import PlotManager
from src.core.Plotting.ProfilePlotter import ProfileMeanEnergyPlotter
from src.core.Validation.ResultAggregator import ResultAggregator
from src.utils import get_et_eta

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

parser = argparse.ArgumentParser(prog="plotter")
parser.add_argument("results_path", type=str)
parser.add_argument("--drive_path", default="data/", type=str, required=False)
parser.add_argument("--percentage", default=None, type=float, required=False)
parser.add_argument("--plot_ringer", action="store_true")
parser.add_argument("--config", default=None, type=str, required=False)
parser.add_argument("--output_dir", default=None, type=str, required=False)
parser.add_argument("--debug", action="store_true")


class PlotterRunner:
    """Runner responsible for batch generating ringer profiles or training metric curves."""

    def __init__(
        self,
        results_path: str,
        drive_path: str = "data/",
        output_dir: Optional[str] = None,
        config_path: Optional[str] = None,
    ) -> None:
        """Initialize the PlotterRunner instance.

        Args:
            results_path: Directory path where training result files are located.
            drive_path: Directory path where raw dataset files are stored.
            output_dir: Target directory for plots (defaults to Plots/<results_folder_name>).
            config_path: Optional path to YAML configuration file.
        """
        self.results_path: Path = Path(results_path)
        self.drive_path: str = drive_path
        self.plot_manager: PlotManager = PlotManager()

        if output_dir:
            self.output_dir: Path = Path(output_dir)
        else:
            target_name = (
                self.results_path.parent.name
                if self.results_path.is_file()
                or self.results_path.suffix in [".pkl", ".npz", ".pt"]
                else (self.results_path.name or "default")
            )
            self.output_dir = Path("Plots") / target_name
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run_ringer_profiles(self, folder_path: str, percentage: float) -> None:
        """Generate mean energy ringer profile plots for all matching npz datasets.

        Args:
            folder_path: Output directory for the generated profile plots.
            percentage: Percentage ratio of rings retained in the dataset.
        """
        data_files = [
            os.path.join(self.drive_path, file)
            for file in os.listdir(self.drive_path)
            if file.endswith(".npz") and file.startswith("mc23_13TeV")
        ]
        dataset = EgammaNpzDataset(self.drive_path, percentage=percentage)
        profile_plotter = ProfileMeanEnergyPlotter()

        for index, file_name in enumerate(data_files):
            data, target, _ = dataset[index]
            et, eta = get_et_eta(file_name)
            context = RegionPlotContext(
                iet=int(et),
                ieta=int(eta),
                output_dir=folder_path,
                data=data,
                target=target,
            )
            profile_plotter.plot(
                context,
                x_rings=data[:, dataset.indexes],
            )

    def run_metrics_plots(
        self, file_path: str, folder_path: str, et: int, eta: int
    ) -> None:
        """Generate training progress and evaluation curve plots from a single result file.

        Args:
            file_path: Path to the result pickle file.
            folder_path: Output directory for saving generated metric figures.
            et: Transverse energy bin index.
            eta: Pseudorapidity bin index.
        """
        self.plot_manager.run_metrics_from_file(file_path, folder_path, et, eta)

    def execute(
        self, plot_ringer: bool = False, percentage: Optional[float] = None
    ) -> None:
        """Execute plotting pipeline across all fold result files in results_path.

        Args:
            plot_ringer: If True, generate ringer profiles instead of metric curves.
            percentage: Required ring percentage when plot_ringer is True.

        Raises:
            ValueError: If plot_ringer is True but percentage is None.
        """
        folder_path = str(self.output_dir)

        if plot_ringer:
            if percentage is None:
                raise ValueError(
                    "You must provide percentage of rings. Ex: --percentage 0.5"
                )
            self.run_ringer_profiles(folder_path, percentage)
            return

        if self.results_path.is_file():
            et, eta = get_et_eta(str(self.results_path))
            if et is None or eta is None:
                log.warning(
                    f"Could not extract et/eta from file {self.results_path}. Skipping."
                )
                return
            self.plot_manager.run_metrics_from_file(
                str(self.results_path), folder_path, int(et), int(eta)
            )
            return

        aggregator = ResultAggregator(self.results_path)
        grouped_results = aggregator.group_and_concat()

        if not grouped_results:
            log.warning(f"No result files found in {self.results_path}.")
            return

        for region_key, df_region in grouped_results.items():
            match = re.search(r"iet(\d+)\.ieta(\d+)", region_key)
            if not match:
                log.warning(
                    f"Could not parse (iet, ieta) from region key '{region_key}'. Skipping."
                )
                continue

            iet, ieta = int(match.group(1)), int(match.group(2))

            if (
                df_region is None
                or df_region.empty
                or "best_sp_value" not in df_region.columns
                or df_region["best_sp_value"].dropna().empty
            ):
                log.warning(f"No metric data found in region {region_key}. Skipping.")
                continue

            log.info(f"Plotting best SP model for region {region_key}...")
            self.plot_manager.run_metrics_for_region(
                df_region, folder_path, iet, ieta
            )


if __name__ == "__main__":
    args = parser.parse_args()
    runner = PlotterRunner(
        results_path=args.results_path,
        drive_path=args.drive_path,
        output_dir=args.output_dir,
        config_path=args.config,
    )
    runner.execute(plot_ringer=args.plot_ringer, percentage=args.percentage)
