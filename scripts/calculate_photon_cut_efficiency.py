import argparse
import logging
import sys
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.Validation.FastPhotonCutEvaluator import (
    FastPhotonCutEvaluator,
    UserKinematicGrid,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parses command line arguments.

    Returns:
        argparse.Namespace with parsed options.
    """
    parser = argparse.ArgumentParser(
        description="Calculate fast photon trigger cut efficiencies across user ET and eta kinematic regions."
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="/home/pmourafr/egamma_master/data/consolidated/",
        help="Path to a directory containing NPZ files or a single NPZ file.",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.npz",
        help="Glob pattern to search for NPZ files when data_path is a directory (default: '*.npz').",
    )
    parser.add_argument(
        "--working_points",
        nargs="+",
        default=["loose", "medium", "tight"],
        help="List of operating points to evaluate (default: loose medium tight).",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="efficiencies_by_region.csv",
        help="Path to save region summary CSV for ModelValidator (default: efficiencies_by_region.csv).",
    )
    return parser.parse_args()


def display_results_table(df_results: "pd.DataFrame", working_points: List[str]) -> None:
    """Displays formatted terminal table of calculated region efficiencies.

    Args:
        df_results: Aggregated DataFrame from FastPhotonCutEvaluator.
        working_points: List of evaluated working points.
    """
    log.info("\n" + "=" * 90)
    log.info("CALCULATED PARTICLE IDENTIFICATION EFFICIENCIES BY KINEMATIC REGION")
    log.info("=" * 90)
    header = f"{'iet':<4} {'ieta':<5} {'ET Region (GeV)':<18} {'|eta| Region':<16} {'Samples':<10}"
    for wp in working_points:
        header += f"{wp.capitalize() + ' Eff':<14}"
    log.info(header)
    log.info("-" * 90)

    for _, row in df_results.iterrows():
        line = (
            f"{int(row['et_bin']):<4} {int(row['eta_bin']):<5} "
            f"{row['et_range']:<18} {row['eta_range']:<16} "
            f"{int(row['total_samples']):<10}"
        )
        for wp in working_points:
            eff_val = row[f"eff_{wp}"]
            passed_val = int(row[f"passed_{wp}"])
            line += f"{eff_val:.4f} ({passed_val})".ljust(14)
        log.info(line)
    log.info("=" * 90 + "\n")


def main() -> None:
    """Executes streaming kinematic region evaluation workflow."""
    args = parse_args()
    data_target = Path(args.data_path)

    if data_target.is_file():
        file_paths = [data_target]
    elif data_target.is_dir():
        file_paths = sorted(list(data_target.glob(args.pattern)))
        if not file_paths:
            file_paths = sorted([f for f in data_target.iterdir() if f.suffix == ".npz"])
    else:
        log.error(f"Provided data_path does not exist: {data_target}")
        sys.exit(1)

    if not file_paths:
        log.error(f"No NPZ files found in: {data_target}")
        sys.exit(1)

    log.info(f"Discovered {len(file_paths)} file(s) to process.")
    grid = UserKinematicGrid()
    evaluator = FastPhotonCutEvaluator(grid=grid)

    log.info("Processing files sequentially and accumulating region statistics...")
    df_results = evaluator.evaluate_files(file_paths, working_points=args.working_points)

    display_results_table(df_results, args.working_points)

    evaluator.export_reference_csv(df_results, args.output_csv)
    log.info(f"Exported reference table to: {args.output_csv}")


if __name__ == "__main__":
    main()
