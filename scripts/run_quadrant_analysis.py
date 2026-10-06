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
    description="Compare models or model vs Athena trigger cut-based selection with quadrant analysis.",
)
parser.add_argument(
    "--mode",
    required=False,
    choices=["model_vs_model", "model_vs_cut"],
    default="model_vs_model",
    help="Comparison mode: 'model_vs_model' or 'model_vs_cut' (default: model_vs_model).",
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
    help="Path to Model 2 configuration YAML (defaults to --config1 for model_vs_model).",
)
parser.add_argument(
    "--data_path2",
    required=False,
    default=None,
    type=str,
    help="Path to Model 2 results directory (required when mode is model_vs_model).",
)
parser.add_argument(
    "--working_point",
    required=False,
    choices=["loose", "medium", "tight"],
    default="loose",
    help="Athena cut working point for model_vs_cut mode (default: loose).",
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
    help="Model decision threshold mode: 'calibrated' for target PD cut, 'default' for 0.5 threshold.",
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
parser.add_argument(
    "--csv_filename",
    required=False,
    default="quadrant_results.csv",
    type=str,
    help="Output CSV filename for tabular results and uncertainties (default: quadrant_results.csv).",
)


def main(args: argparse.Namespace) -> None:
    """Executes the quadrant analysis comparison pipeline.

    Args:
        args: Parsed command-line arguments.
    """
    if args.mode == "model_vs_model":
        if args.data_path2 is None:
            parser.error("--data_path2 is required when --mode is 'model_vs_model'.")
        config2_path = args.config2 if args.config2 is not None else args.config1
    else:
        config2_path = None

    analyzer = QuadrantAnalyzer(
        config_path1=args.config1,
        data_path1=args.data_path1,
        config_path2=config2_path,
        data_path2=args.data_path2,
        manifest_path=args.manifest_path,
        threshold_mode=args.threshold_mode,
        efficiencies_csv=args.efficiencies_csv,
        default_target_pd=args.target_pd,
        mode=args.mode,
        working_point=args.working_point,
    )
    output_dir = analyzer.get_output_dir(args.output_dir)

    log.info("=" * 78)
    log.info("STARTING QUADRANT ANALYSIS PIPELINE")
    log.info("=" * 78)
    log.info(f"Mode:                {args.mode}")
    log.info(f"Strategy 1 Config:   {args.config1}")
    log.info(f"Strategy 1 Results:  {args.data_path1}")
    if args.mode == "model_vs_model":
        log.info(f"Strategy 2 Config:   {config2_path}")
        log.info(f"Strategy 2 Results:  {args.data_path2}")
    else:
        log.info(f"Strategy 2 Cut WP:   Athena Fast Photon {args.working_point.capitalize()}")
    log.info(f"Comparison Tag:      {analyzer.comparison_tag}")
    log.info(f"Threshold Mode:      {args.threshold_mode}")
    if args.threshold_mode == "calibrated":
        log.info(f"Target PD Cut:       {args.target_pd}")
        if args.efficiencies_csv:
            log.info(f"Efficiencies CSV:    {args.efficiencies_csv}")
    log.info(f"Output Directory:    {output_dir}")
    log.info(f"Plot Format:         {args.file_format}")
    log.info("-" * 78)

    regional_results = analyzer.run()
    if not regional_results:
        log.warning("No overlapping regions found. Skipping plot generation. Exiting.")
        return

    log.info("-" * 128)
    log.info(f"Successfully evaluated {len(regional_results)} kinematic regions.")
    log.info("-" * 128)
    header = (
        f"{'Region':<12} | "
        f"{'S1 Pd':<7} {'S1 BgEff':<8} {'S1 Pf':<7} {'S1 SP':<7} | "
        f"{'S2 Pd':<7} {'S2 BgEff':<8} {'S2 Pf':<7} {'S2 SP':<7} | "
        f"{'Both Right':<14} | {'S1 Adv':<12} | {'S2 Adv':<12} | {'Both Wrong':<12} | {'McNemar p':<10}"
    )
    log.info(header)
    log.info("-" * 144)

    for res in regional_results:
        m = res.overall_metrics
        s1 = res.metrics_strategy1
        s2 = res.metrics_strategy2
        region_label = f"iet{res.iet}.ieta{res.ieta}"
        log.info(
            f"{region_label:<12} | "
            f"{s1.pd:<7.4f} {s1.bg_eff:<8.4f} {s1.pf:<7.4f} {s1.sp:<7.4f} | "
            f"{s2.pd:<7.4f} {s2.bg_eff:<8.4f} {s2.pf:<7.4f} {s2.sp:<7.4f} | "
            f"{m.both_correct} ({m.both_correct_ratio:.1%})".ljust(14) + " | " +
            f"{m.model1_only_correct} ({m.model1_only_correct_ratio:.1%})".ljust(12) + " | " +
            f"{m.model2_only_correct} ({m.model2_only_correct_ratio:.1%})".ljust(12) + " | " +
            f"{m.both_wrong} ({m.both_wrong_ratio:.1%})".ljust(12) + " | " +
            f"{m.mcnemar_p_value:.3e}"
        )
    log.info("-" * 144)

    global_result = analyzer.compute_global_result(regional_results)
    if global_result is not None:
        gm = global_result.overall_metrics
        gs1 = global_result.metrics_strategy1
        gs2 = global_result.metrics_strategy2
        log.info(
            f"{'GLOBAL':<12} | "
            f"{gs1.pd:<7.4f} {gs1.bg_eff:<8.4f} {gs1.pf:<7.4f} {gs1.sp:<7.4f} | "
            f"{gs2.pd:<7.4f} {gs2.bg_eff:<8.4f} {gs2.pf:<7.4f} {gs2.sp:<7.4f} | "
            f"{gm.both_correct} ({gm.both_correct_ratio:.1%})".ljust(14) + " | " +
            f"{gm.model1_only_correct} ({gm.model1_only_correct_ratio:.1%})".ljust(12) + " | " +
            f"{gm.model2_only_correct} ({gm.model2_only_correct_ratio:.1%})".ljust(12) + " | " +
            f"{gm.both_wrong} ({gm.both_wrong_ratio:.1%})".ljust(12) + " | " +
            f"{gm.mcnemar_p_value:.3e}"
        )
        log.info("-" * 144)

    log.info(f"Generating trigger shower shape histograms and efficiency curves under: {output_dir}")
    plotter = QuadrantPlotter()
    plot_results = list(regional_results)
    if global_result is not None:
        plot_results.append(global_result)

    saved_paths = plotter.plot(
        results=plot_results,
        output_dir=output_dir,
        file_format=args.file_format,
    )

    total_figures = sum(len(paths) for paths in saved_paths.values())
    log.info(f"Completed! Total generated figures: {total_figures} under {output_dir}")
    log.info("-" * 78)
    log.info(f"Saving tabular results with strategy metrics under: {output_dir}")
    csv_path = analyzer.save_results_table(
        regional_results=regional_results,
        output_dir=output_dir,
        filename=args.csv_filename,
    )
    log.info(f"Tabular results saved to: {csv_path}")
    log.info("=" * 78)


if __name__ == "__main__":
    cli_args = parser.parse_args()
    main(cli_args)
