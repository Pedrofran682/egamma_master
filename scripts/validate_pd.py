import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.Validation.ModelValidator import ModelValidator

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

parser = argparse.ArgumentParser(prog="NeuralRingerValidator")
parser.add_argument(
    "--config", required=True, type=str, help="Path to configuration YAML."
)
parser.add_argument(
    "--data_path", required=True, type=str, help="Path to processed data directory."
)
parser.add_argument(
    "--efficiencies_csv",
    required=False,
    default=None,
    type=str,
    help="Optional path to reference operating point CSV table (if omitted, uses --target_pd).",
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
    default=None,
    type=str,
    help="Custom path to output plots (defaults to Plots/<results_folder_name>).",
)


def main(args: argparse.Namespace) -> None:
    """Run model validation pipeline comparing neural models against baseline threshold cuts.

    Args:
        args: Parsed command-line arguments.
    """
    validator = ModelValidator(
        config_path=args.config,
        data_path=args.data_path,
        efficiencies_csv=args.efficiencies_csv,
        default_target_pd=args.target_pd,
        output_dir=args.output_dir,
    )
    validator.run()


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
