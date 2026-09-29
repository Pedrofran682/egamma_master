import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.Validation.RegionDataDistributionAnalyzer import RegionDataDistributionAnalyzer

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

parser = argparse.ArgumentParser(
    prog="PlotRegionDataDistribution",
    description="Plot the total amount of data by class 0 or 1 for each ET and eta region.",
)
parser.add_argument(
    "--config",
    required=True,
    type=str,
    help="Path to training YAML configuration file.",
)
parser.add_argument(
    "--data_path",
    required=False,
    default=None,
    type=str,
    help="Path to processed data directory (overrides config drive_path).",
)
parser.add_argument(
    "--output_dir",
    required=False,
    default=None,
    type=str,
    help="Custom path to output directory for plots and summary CSV.",
)
parser.add_argument(
    "--format",
    required=False,
    default="pdf",
    choices=["pdf", "png"],
    help="File format for generated figures (default: pdf).",
)


def main(args: argparse.Namespace) -> None:
    """Orchestrates regional class distribution analysis and 2D heatmap generation.

    Args:
        args: Parsed command-line arguments.
    """
    analyzer = RegionDataDistributionAnalyzer(
        config_path=args.config,
        data_path=args.data_path,
        output_dir=args.output_dir,
    )
    analyzer.run(file_format=args.format)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
