import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.Validation.FastPhotonCutEvaluator import (
    FastPhotonCutEvaluator,
    UserKinematicGrid,
)
from src.utils import get_et_eta

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


def parse_args(args: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parses command line arguments.

    Args:
        args: Optional sequence of argument strings. Defaults to sys.argv[1:].

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
        "--output_dir",
        type=str,
        default=None,
        help="Destination directory for output files (default: None, saves to current directory or path in --output_csv).",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="efficiencies_by_region.csv",
        help="Path to save region summary CSV for ModelValidator (default: efficiencies_by_region.csv).",
    )
    parser.add_argument(
        "--et",
        "--iet",
        dest="et",
        type=int,
        default=None,
        help="Filter to specific ET bin index.",
    )
    parser.add_argument(
        "--eta",
        "--ieta",
        dest="eta",
        type=int,
        default=None,
        help="Filter to specific eta bin index.",
    )
    return parser.parse_args(args)


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

    if args.et is not None or args.eta is not None:
        filtered_files = []
        for fp in file_paths:
            fet, feta = get_et_eta(str(fp))
            if fet is not None and feta is not None:
                if args.et is not None and int(fet) != args.et:
                    continue
                if args.eta is not None and int(feta) != args.eta:
                    continue
            filtered_files.append(fp)
        file_paths = filtered_files
        if not file_paths:
            log.warning(f"No files matched region filter et={args.et}, eta={args.eta}.")

    log.info(f"Discovered {len(file_paths)} file(s) to process.")
    grid = UserKinematicGrid()
    evaluator = FastPhotonCutEvaluator(grid=grid)

    log.info("Processing files sequentially and accumulating region statistics...")
    df_results = evaluator.evaluate_files(file_paths, working_points=args.working_points)

    if args.et is not None:
        df_results = df_results[df_results["et_bin"] == args.et]
    if args.eta is not None:
        df_results = df_results[df_results["eta_bin"] == args.eta]

    display_results_table(df_results, args.working_points)

    output_csv_path = Path(args.output_csv)
    if args.output_dir is not None:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        output_csv_path = out_dir / output_csv_path.name

    evaluator.export_reference_csv(df_results, output_csv_path)
    log.info(f"Exported reference table to: {output_csv_path}")


if __name__ == "__main__":
    main()
