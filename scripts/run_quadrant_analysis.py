import argparse
import logging
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from src.core.Plotting.QuadrantPlotter import QuadrantPlotter
from src.core.Validation.QuadrantAnalyzer import QuadrantAnalyzer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("QuadrantAnalysis")

parser = argparse.ArgumentParser(
    prog="QuadrantAnalysis",
    description="Compare two classification models using holdout test sets and quadrant analysis.",
)
parser.add_argument(
    "--config1",
    required=True,
    type=str,
    help="Path to Model 1 configuration YAML.",
)
parser.add_argument(
    "--data_path1",
    required=True,
    type=str,
    help="Path to Model 1 results directory (containing .pkl files and split_manifest.json).",
)
parser.add_argument(
    "--config2",
    required=False,
    default=None,
    type=str,
    help="Path to Model 2 configuration YAML (defaults to --config1 if omitted).",
)
parser.add_argument(
    "--data_path2",
    required=True,
    type=str,
    help="Path to Model 2 results directory (containing .pkl files).",
)
parser.add_argument(
    "--manifest_path",
    required=False,
    default=None,
    type=str,
    help="Optional path to explicit split manifest JSON file.",
)
parser.add_argument(
    "--threshold_mode",
    required=False,
    choices=["calibrated", "default"],
    default="calibrated",
    help="Decision threshold mode: 'calibrated' for target PD cut, 'default' for standard 0.5 threshold.",
)
parser.add_argument(
    "--efficiencies_csv",
    required=False,
    default=None,
    type=str,
    help="Optional path to reference operating point CSV table.",
)
parser.add_argument(
    "--target_pd",
    required=False,
    default=0.9424,
    type=float,
    help="Target signal efficiency value (default: 0.9424).",
)
parser.add_argument(
    "--output_dir",
    required=False,
    default="Plots/quandrantic_analysis",
    type=str,
    help="Destination directory for quadrant plots (default: Plots/quandrantic_analysis).",
)
parser.add_argument(
    "--file_format",
    required=False,
    choices=["pdf", "png"],
    default="pdf",
    help="Graphic output format ('pdf' or 'png', default: 'pdf').",
)


def main(args: argparse.Namespace) -> None:
    """Executes the quadrant analysis comparison pipeline.

    Args:
        args: Parsed command-line arguments.
    """
    config2_path = args.config2 if args.config2 is not None else args.config1

    log.info("=" * 70)
    log.info("STARTING QUADRANT ANALYSIS MODEL COMPARISON")
    log.info("=" * 70)
    log.info(f"Model 1 Config:      {args.config1}")
    log.info(f"Model 1 Results:     {args.data_path1}")
    log.info(f"Model 2 Config:      {config2_path}")
    log.info(f"Model 2 Results:     {args.data_path2}")
    log.info(f"Threshold Mode:      {args.threshold_mode}")
    if args.threshold_mode == "calibrated":
        log.info(f"Target PD Cut:       {args.target_pd}")
        if args.efficiencies_csv:
            log.info(f"Efficiencies CSV:    {args.efficiencies_csv}")
    log.info(f"Output Plot Dir:     {args.output_dir}")
    log.info(f"Plot Format:         {args.file_format}")
    log.info("-" * 70)

    analyzer = QuadrantAnalyzer(
        config_path1=args.config1,
        data_path1=args.data_path1,
        config_path2=config2_path,
        data_path2=args.data_path2,
        manifest_path=args.manifest_path,
        threshold_mode=args.threshold_mode,
        efficiencies_csv=args.efficiencies_csv,
        default_target_pd=args.target_pd,
    )

    regional_results = analyzer.run()
    if not regional_results:
        log.warning(
            "No overlapping regions with trained models for BOTH Model 1 and Model 2 were found. "
            "Skipping plot generation. Exiting."
        )
        return

    log.info("-" * 70)
    log.info(f"Successfully evaluated {len(regional_results)} kinematic regions.")
    log.info("-" * 70)
    log.info(
        f"{'Region':<16} | {'Events':<8} | {'Both Right':<14} | {'M1 Adv':<12} | {'M2 Adv':<12} | {'Both Wrong':<12} | {'McNemar p':<10}"
    )
    log.info("-" * 96)
    for res in regional_results:
        m = res.overall_metrics
        region_label = f"iet{res.iet}.ieta{res.ieta}"
        log.info(
            f"{region_label:<16} | {m.total_events:<8} | "
            f"{m.both_correct} ({m.both_correct_ratio:.1%})".ljust(14) + " | " +
            f"{m.model1_only_correct} ({m.model1_only_correct_ratio:.1%})".ljust(12) + " | " +
            f"{m.model2_only_correct} ({m.model2_only_correct_ratio:.1%})".ljust(12) + " | " +
            f"{m.both_wrong} ({m.both_wrong_ratio:.1%})".ljust(12) + " | " +
            f"{m.mcnemar_p_value:.3e}"
        )
    log.info("-" * 96)

    global_result = analyzer.compute_global_result(regional_results)
    if global_result is not None:
        gm = global_result.overall_metrics
        log.info(
            f"{'GLOBAL':<16} | {gm.total_events:<8} | "
            f"{gm.both_correct} ({gm.both_correct_ratio:.1%})".ljust(14) + " | " +
            f"{gm.model1_only_correct} ({gm.model1_only_correct_ratio:.1%})".ljust(12) + " | " +
            f"{gm.model2_only_correct} ({gm.model2_only_correct_ratio:.1%})".ljust(12) + " | " +
            f"{gm.both_wrong} ({gm.both_wrong_ratio:.1%})".ljust(12) + " | " +
            f"{gm.mcnemar_p_value:.3e}"
        )
        log.info("-" * 96)

    all_results = regional_results + ([global_result] if global_result is not None else [])

    log.info(f"Generating plots under: {args.output_dir}")
    plotter = QuadrantPlotter()
    saved_paths = plotter.plot(
        results=all_results,
        output_dir=args.output_dir,
        file_format=args.file_format,
    )

    total_figures = sum(len(paths) for paths in saved_paths.values())
    log.info(f"Completed! Total generated figures: {total_figures} under {args.output_dir}")
    log.info("=" * 70)


if __name__ == "__main__":
    cli_args = parser.parse_args()
    main(cli_args)
