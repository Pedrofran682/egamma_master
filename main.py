import argparse
import logging
import os
from datetime import datetime
from logging.config import fileConfig

import yaml

from src.core.Trainers.NeuralRingerTrainer import NeuralRingerTrainer
from src.Parser.NeuralRingerTrainerConfiguration import NeuralRingerTrainerConfiguration

CONFIG_FILE = "logging.ini"
LOG_DIR = "log"
os.makedirs(LOG_DIR, exist_ok=True)
log_filename = f"{LOG_DIR}/TrainerRunner_{datetime.now().strftime('%Y%m%d%H%M%S')}.log"
fileConfig(
    CONFIG_FILE,
    defaults={"log_file_path": log_filename},
    disable_existing_loggers=False,
)
log = logging.getLogger(__name__)

def build_parser() -> argparse.ArgumentParser:
    """Build and configure the command-line argument parser.

    Returns:
        Configured ArgumentParser instance.
    """
    parser = argparse.ArgumentParser(prog="TrainerRunner")
    parser.add_argument("--config", required=True, type=str, help="Path to YAML training configuration file.")
    parser.add_argument(
        "--results_path",
        required=False,
        default=None,
        type=str,
        help="Optional path to output results directory to resume or store outputs.",
    )
    return parser


def main(args: argparse.Namespace) -> None:
    """Load YAML training configuration and execute NeuralRingerTrainer.

    Args:
        args: Parsed command-line arguments containing config path and optional results path.
    """
    with open(args.config, "r") as file:
        yaml_data = yaml.safe_load(file)
        if args.results_path is not None:
            yaml_data["results_folder_path"] = args.results_path
        config_instance = NeuralRingerTrainerConfiguration.model_validate(yaml_data)
        trainer = NeuralRingerTrainer(
            config_instance,
        )
        trainer.run()


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    log.info(args)
    main(args)

