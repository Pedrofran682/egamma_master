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


def main(args: argparse.Namespace) -> None:
    """Run model validation pipeline comparing neural models against baseline threshold cuts.

    Args:
        args: Parsed command-line arguments containing config and data_path.
    """
    validator = ModelValidator(config_path=args.config, data_path=args.data_path)
    validator.run()


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
