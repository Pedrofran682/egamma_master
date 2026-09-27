import argparse
import logging
import sys
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.Datasets.RegionDataConsolidator import RegionDataConsolidator

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parses command line arguments for merging regional NPZ dataset partitions.

    Returns:
        argparse.Namespace with parsed command-line options.
    """
    parser = argparse.ArgumentParser(
        description="Merge partitioned calorimeter NPZ dataset files into consolidated region files."
    )
    parser.add_argument(
        "--input_dir",
        "-i",
        type=str,
        default="./data",
        help="Source directory containing partitioned NPZ files (default: './data').",
    )
    parser.add_argument(
        "--output_dir",
        "-o",
        type=str,
        default="./data/consolidated",
        help="Destination directory for consolidated NPZ files (default: './data/consolidated').",
    )
    parser.add_argument(
        "--workers",
        "-w",
        type=int,
        default=4,
        help="Number of parallel worker processes. Defaults to all available CPU cores.",
    )
    parser.add_argument(
        "--no_compress",
        action="store_true",
        help="Save uncompressed NPZ archives instead of np.savez_compressed.",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default=r"\.et(\d+)\.eta(\d+).*\.npz$",
        help="Regex pattern matching et and eta indices from filenames.",
    )
    parser.add_argument(
        "--ignore_pattern",
        type=str,
        default=r"\.sys\.v\d+",
        help="Regex pattern identifying files to ignore.",
    )
    return parser.parse_args()


def concatenate_npz_by_group(
    base_directory: str | Path,
    output_directory: str | Path,
    max_workers: int | None = None,
    compressed: bool = True,
) -> List[Path]:
    """Consolidates NPZ partition files by (et, eta) kinematic groups.

    Args:
        base_directory: Directory containing partitioned NPZ files.
        output_directory: Directory where consolidated NPZ files will be saved.
        max_workers: Number of parallel worker processes.
        compressed: Whether to compress output archives.

    Returns:
        List of generated consolidated NPZ file paths.
    """
    consolidator = RegionDataConsolidator(
        base_dir=base_directory,
        output_dir=output_directory,
        max_workers=max_workers,
        compressed=compressed,
    )
    return consolidator.run()


def concatenate_npz_by_group_v2(
    base_directory: str | Path,
    output_directory: str | Path,
    max_workers: int | None = None,
    compressed: bool = True,
) -> List[Path]:
    """Compatibility alias for concatenate_npz_by_group.

    Args:
        base_directory: Directory containing partitioned NPZ files.
        output_directory: Directory where consolidated NPZ files will be saved.
        max_workers: Number of parallel worker processes.
        compressed: Whether to compress output archives.

    Returns:
        List of generated consolidated NPZ file paths.
    """
    return concatenate_npz_by_group(
        base_directory=base_directory,
        output_directory=output_directory,
        max_workers=max_workers,
        compressed=compressed,
    )


def main() -> None:
    """CLI entrypoint for consolidating regional NPZ partitions."""
    args = parse_args()
    log.info("Starting NPZ dataset consolidation from %s to %s", args.input_dir, args.output_dir)

    consolidator = RegionDataConsolidator(
        base_dir=args.input_dir,
        output_dir=args.output_dir,
        name_pattern=args.pattern,
        ignore_pattern=args.ignore_pattern,
        max_workers=args.workers,
        compressed=not args.no_compress,
    )
    outputs = consolidator.run()
    log.info("Finished consolidation. Generated %d files.", len(outputs))


if __name__ == "__main__":
    main()
