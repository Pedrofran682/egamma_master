import argparse
import logging
import os
import re
import sys
from datetime import datetime
from logging.config import fileConfig
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.Plotting.PlotManager import PlotManager
from src.core.Validation.ResultAggregator import ResultAggregator
from src.utils import get_et_eta

CONFIG_FILE = "logging.ini"
LOG_DIR = "log"
log = logging.getLogger(__name__)


def _configure_logging() -> None:
    """Configures file and console logging when running as a CLI script."""
    os.makedirs(LOG_DIR, exist_ok=True)
    log_filename = f"{LOG_DIR}/plotter_{datetime.now().strftime('%Y%m%d%H%M%S')}.log"
    fileConfig(
        CONFIG_FILE,
        defaults={"log_file_path": log_filename},
        disable_existing_loggers=False,
    )


parser = argparse.ArgumentParser(prog="plotter")
parser.add_argument("results_path", type=str)
parser.add_argument("--drive_path", default="data/", type=str, required=False)
parser.add_argument("--config", default=None, type=str, required=False)
parser.add_argument("--output_dir", default=None, type=str, required=False)
parser.add_argument(
    "--et",
    "--iet",
    dest="et",
    default=None,
    type=int,
    required=False,
    help="Filter to specific ET bin index.",
)
parser.add_argument(
    "--eta",
    "--ieta",
    dest="eta",
    default=None,
    type=int,
    required=False,
    help="Filter to specific eta bin index.",
)
parser.add_argument(
    "--lang",
    default="en",
    choices=["en", "pt", "both"],
    help="Target language for plot labels and titles (en, pt, or both).",
)
parser.add_argument("--debug", action="store_true")


class PlotterRunner:
    """Runner responsible for batch generating training metric curves."""

    def __init__(
        self,
        results_path: str,
        drive_path: str = "data/",
        output_dir: Optional[str] = None,
        config_path: Optional[str] = None,
        target_et: Optional[int] = None,
        target_eta: Optional[int] = None,
        language: str = "en",
    ) -> None:
        """Initialize the PlotterRunner instance.

        Args:
            results_path: Directory path where training result files are located.
            drive_path: Directory path where raw dataset files are stored.
            output_dir: Target directory for plots (defaults to Plots/<results_folder_name>).
            config_path: Optional path to YAML configuration file.
            target_et: Optional ET bin index filter.
            target_eta: Optional eta bin index filter.
            language: Target localization language ('en', 'pt', or 'both').
        """
        self.results_path: Path = Path(results_path)
        self.drive_path: str = drive_path
        self.config_path: Optional[str] = config_path
        self.target_et: Optional[int] = target_et
        self.target_eta: Optional[int] = target_eta
        self.language: str = language
        self.plot_manager: PlotManager = PlotManager(language=language)

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

    def execute(self) -> None:
        """Execute metric plotting pipeline across all fold result files in results_path."""
        folder_path = str(self.output_dir)

        if self.results_path.is_file():
            et, eta = get_et_eta(str(self.results_path))
            if et is None or eta is None:
                log.warning(
                    f"Could not extract et/eta from file {self.results_path}. Skipping."
                )
                return
            if self.target_et is not None and int(et) != self.target_et:
                return
            if self.target_eta is not None and int(eta) != self.target_eta:
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
            if self.target_et is not None and iet != self.target_et:
                continue
            if self.target_eta is not None and ieta != self.target_eta:
                continue

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


def main(args: argparse.Namespace) -> None:
    """CLI entrypoint for training metric plot generation."""
    _configure_logging()
    runner = PlotterRunner(
        results_path=args.results_path,
        drive_path=args.drive_path,
        output_dir=args.output_dir,
        config_path=args.config,
        target_et=args.et,
        target_eta=args.eta,
        language=args.lang,
    )
    runner.execute()


if __name__ == "__main__":
    cli_args = parser.parse_args()
    main(cli_args)
