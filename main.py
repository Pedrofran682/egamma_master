import argparse
import logging
import os
from datetime import datetime
from logging.config import fileConfig
from pathlib import Path

import yaml

from src.core.Handlers.ProcessLifecycleManager import ProcessLifecycleManager
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
lifecycle_manager = ProcessLifecycleManager(log)
lifecycle_manager.install()


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
    parser.add_argument(
        "--split_manifest",
        required=True,
        type=str,
        help="Path or filename of split manifest in ./results (pass 'None' to create inside output folder).",
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
        if args.split_manifest is not None and args.split_manifest.strip().lower() not in ["none", ""]:
            manifest_path = Path(args.split_manifest)
            if not manifest_path.is_absolute() and len(manifest_path.parts) == 1:
                manifest_path = Path("results") / manifest_path
            yaml_data["split_manifest_path"] = str(manifest_path)
        else:
            yaml_data["split_manifest_path"] = None
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

