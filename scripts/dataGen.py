import argparse
import logging
import sys
from pathlib import Path
from typing import List, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.Datasets.RootDatasetGenerator import (
    DEFAULT_ETA_BINS,
    DEFAULT_ET_BINS,
    DEFAULT_TARGET_COLUMNS,
    RootDatasetGenerator,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)

DEFAULT_SAMPLE_DIRS: List[str] = [
    "user.pmourafr.mc23_valid.801278.Py8EG_A14NNPDF23LO_perf_JF17.re.photonRinger184230042026_XYZ.root.tgz",
    "user.pmourafr.mc23_valid.801279.Py8EG_A14NNPDF23LO_perf_JF35.re.photonRinger184230042026_XYZ.root.tgz",
    "user.pmourafr.mc23_valid.801280.Py8EG_A14NNPDF23LO_perf_JF50.re.photonRinger184230042026_XYZ.root.tgz",
    "user.pmourafr.mc23_valid.801650.Py8_gammajet_frag_DP17_35_FullS.photonRinger183230042026_XYZ.root.tgz",
    "user.pmourafr.mc23_valid.801651.Py8_gammajet_frag_DP35_50_FullS.photonRinger085801052026_XYZ.root.tgz",
    "user.pmourafr.mc23_valid.801651.Py8_gammajet_frag_DP35_50_FullS.photonRinger183230042026_XYZ.root.tgz",
    "user.pmourafr.mc23_valid.801652.Py8_gammajet_frag_DP50_70_FullS.photonRinger010501052026_XYZ.root.tgz",
    "user.pmourafr.mc23_valid.801652.Py8_gammajet_frag_DP50_70_FullS.photonRinger090501052026_XYZ.root.tgz",
    "user.pmourafr.mc23_valid.801652.Py8_gammajet_frag_DP50_70_FullS.photonRinger183230042026_XYZ.root.tgz",
    "user.pmourafr.mc23_valid.801653.Py8_gammajet_frag_DP70_140_Full.photonRinger183230042026_XYZ.root.tgz",
]


def parse_args() -> argparse.Namespace:
    """Parses command line arguments for ROOT dataset partitioning.

    Returns:
        argparse.Namespace with parsed CLI arguments.
    """
    parser = argparse.ArgumentParser(
        description="Extract and partition ATLAS calorimeter events from ROOT files into NPZ datasets."
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        default="/eos/user/p/pmourafr/RootFiles",
        help="Base directory containing unpacked ROOT sample directories.",
    )
    parser.add_argument(
        "--output_dir",
        "-o",
        type=str,
        default="/eos/user/p/pmourafr/egamma_master/data",
        help="Destination directory for generated NPZ files.",
    )
    parser.add_argument(
        "--tree_name",
        type=str,
        default="run_450000/HLT/EgammaMon/summary/events",
        help="TTree path within the ROOT files.",
    )
    parser.add_argument(
        "--vector_col",
        type=str,
        default="trig_L2_calo_rings",
        help="Branch name containing calorimeter ring vector data.",
    )
    parser.add_argument(
        "--splits",
        type=int,
        default=2,
        help="Number of batches to split input files per dataset.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of worker threads for uproot ROOT file decompression.",
    )
    parser.add_argument(
        "--compress",
        action="store_true",
        help="Compress generated NPZ archives.",
    )
    parser.add_argument(
        "--input_dirs",
        nargs="*",
        default=None,
        help="Explicit list of relative or absolute sample directory names.",
    )
    return parser.parse_args()


def process_root_files(
    input_dir: str | Path,
    output_dir: str | Path,
    tree_name: str,
    vector_col: str,
    eta_bins: Sequence[float],
    et_bins: Sequence[float],
    target_columns: List[str],
    file_prefix: str,
    num_splits: int = 2,
    uproot_workers: int = 4,
    compressed: bool = False,
) -> List[Path]:
    """Compatibility function to process ROOT files in batches and save binned NPZ files.

    Args:
        input_dir: Directory containing ROOT files.
        output_dir: Base directory where output dataset folder is created.
        tree_name: Name of TTree inside the ROOT files.
        vector_col: Column containing vector calorimeter rings.
        eta_bins: Bin boundaries for pseudorapidity.
        et_bins: Bin boundaries for transverse energy in MeV.
        target_columns: Feature names to extract from TTree.
        file_prefix: Prefix identifier for the output dataset.
        num_splits: Number of batches to split input files into.
        uproot_workers: Parallel thread count for uproot reading.
        compressed: Whether to compress the output NPZ files.

    Returns:
        List of generated NPZ file paths.
    """
    generator = RootDatasetGenerator(
        tree_name=tree_name,
        vector_col=vector_col,
        target_columns=target_columns,
        eta_bins=eta_bins,
        et_bins=et_bins,
        num_splits=num_splits,
        uproot_workers=uproot_workers,
        compressed=compressed,
    )
    return generator.process_directory(
        input_dir=input_dir,
        output_dir=output_dir,
        file_prefix=file_prefix,
    )


def main() -> None:
    """CLI entrypoint for processing ROOT sample directories."""
    args = parse_args()
    base_path = Path(args.base_dir)
    target_dirs = args.input_dirs if args.input_dirs is not None else DEFAULT_SAMPLE_DIRS

    generator = RootDatasetGenerator(
        tree_name=args.tree_name,
        vector_col=args.vector_col,
        target_columns=DEFAULT_TARGET_COLUMNS,
        eta_bins=DEFAULT_ETA_BINS,
        et_bins=DEFAULT_ET_BINS,
        num_splits=args.splits,
        uproot_workers=args.workers,
        compressed=args.compress,
    )

    for sample_dir in target_dirs:
        dir_path = Path(sample_dir)
        full_path = dir_path if dir_path.is_absolute() else base_path / dir_path

        prefix = dir_path.name
        if "user.pmourafr." in prefix:
            prefix = prefix.split("user.pmourafr.")[1]
        if ".root.tgz" in prefix:
            prefix = prefix.split(".root.tgz")[0]

        generator.process_directory(
            input_dir=full_path,
            output_dir=args.output_dir,
            file_prefix=prefix,
        )


if __name__ == "__main__":
    main()
