import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.Plotting.Localization import BilingualPlotOrchestrator
from src.core.Plotting.RegionDistributionPlotter import RegionDistributionPlotter
from src.core.Validation.RegionDataDistributionAnalyzer import RegionDataDistributionAnalyzer

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
parser.add_argument(
    "--et",
    "--iet",
    dest="et",
    required=False,
    default=None,
    type=int,
    help="Filter to specific ET bin index.",
)
parser.add_argument(
    "--eta",
    "--ieta",
    dest="eta",
    required=False,
    default=None,
    type=int,
    help="Filter to specific eta bin index.",
)
parser.add_argument(
    "--lang",
    required=False,
    default="en",
    choices=["en", "pt", "both"],
    help="Target language for plot labels and titles (en, pt, or both).",
)


def main(args: argparse.Namespace) -> None:
    """Orchestrates regional class distribution analysis and 2D heatmap generation.

    Args:
        args: Parsed command-line arguments.
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    analyzer = RegionDataDistributionAnalyzer(
        config_path=args.config,
        data_path=args.data_path,
        output_dir=args.output_dir,
        target_et=args.et,
        target_eta=args.eta,
    )
    distribution_dataframe = analyzer.collect_distribution()
    if distribution_dataframe.empty:
        log.warning("No data found to plot.")
        return

    csv_summary_path = analyzer.output_dir / "region_data_distribution.csv"
    distribution_dataframe.to_csv(csv_summary_path, index=False)
    log.info(f"Saved distribution summary to {csv_summary_path}")

    orchestrator = BilingualPlotOrchestrator(language=args.lang)

    def _render_language(lang: str, resolved_dir: Path) -> None:
        plotter = RegionDistributionPlotter(
            et_intervals=analyzer.plotter.et_intervals,
            eta_intervals=analyzer.plotter.eta_intervals,
            language=lang,
        )
        generated_plots = plotter.plot(
            region_distribution_dataframe=distribution_dataframe,
            output_dir=resolved_dir,
            file_format=args.format,
            subfolder="",
        )
        for plot_name, plot_path in generated_plots.items():
            log.info(f"Generated plot '{plot_name}' [{lang}]: {plot_path}")

    orchestrator.orchestrate(analyzer.output_dir, _render_language)


if __name__ == "__main__":
    cli_args = parser.parse_args()
    main(cli_args)
