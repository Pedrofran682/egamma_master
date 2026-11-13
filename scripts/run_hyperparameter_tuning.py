import argparse
import logging
import os
import sys
from datetime import datetime
from logging.config import fileConfig
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.Handlers.ProcessLifecycleManager import ProcessLifecycleManager
from src.core.Tuning.GridSearchTuner import GridSearchTuner
from src.core.Tuning.TuningConfiguration import TuningConfiguration

CONFIG_FILE = "logging.ini"
LOG_DIR = "log"
os.makedirs(LOG_DIR, exist_ok=True)
log_filename = f"{LOG_DIR}/TuningRunner_{datetime.now().strftime('%Y%m%d%H%M%S')}.log"
if os.path.exists(CONFIG_FILE):
    fileConfig(
        CONFIG_FILE,
        defaults={"log_file_path": log_filename},
        disable_existing_loggers=False,
    )
log = logging.getLogger(__name__)
lifecycle_manager = ProcessLifecycleManager(log)
lifecycle_manager.install()


def build_parser() -> argparse.ArgumentParser:
    """Builds and configures the command-line argument parser for hyperparameter tuning.

    Returns:
        Configured ArgumentParser instance.
    """
    parser = argparse.ArgumentParser(prog="HyperparameterTuningRunner")
    parser.add_argument(
        "--tuning_config",
        required=True,
        type=str,
        help="Path to YAML hyperparameter tuning configuration file.",
    )
    parser.add_argument(
        "--output_dir",
        required=False,
        default=None,
        type=str,
        help="Optional base destination directory for trial outputs.",
    )
    parser.add_argument(
        "--split_manifest",
        required=False,
        default=None,
        type=str,
        help="Optional path to pre-existing split_manifest.json to enforce exact cross-validation splits.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Activates fast debug mode (epochs=2, n_initializations=1).",
    )
    return parser


def main(args: argparse.Namespace) -> None:
    """Executes the hyperparameter tuning workflow.

    Args:
        args: Parsed command-line arguments.
    """
    tuning_config = TuningConfiguration.from_yaml(args.tuning_config)
    if args.output_dir is not None:
        tuning_config.output_dir = args.output_dir
    if args.split_manifest is not None:
        tuning_config.split_manifest_path = args.split_manifest
    if args.debug:
        tuning_config.overrides["debug"] = True

    tuner = GridSearchTuner(tuning_config)
    results_df = tuner.run()

    log.info("\nTop 5 Hyperparameter Configurations:")
    top_cols = [c for c in ["trial_name", "optimizer", "mean_sp", "std_sp", "best_sp"] if c in results_df.columns]
    log.info(f"\n{results_df[top_cols].head(5).to_string(index=False)}")


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    log.info(args)
    main(args)
