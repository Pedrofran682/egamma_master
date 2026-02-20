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
fileConfig(CONFIG_FILE, defaults={"log_file_path": log_filename})
log = logging.getLogger(__name__)

parser = argparse.ArgumentParser(prog="TrainerRunner")
parser.add_argument("--config", required=True, type=str)


def main(args: argparse.Namespace):

    with open(args.config, "r") as file:
        yaml_data = yaml.safe_load(file)
        config_instance = NeuralRingerTrainerConfiguration.model_validate(yaml_data)
    if bool(args.debug):
        trainer = NeuralRingerTrainer(
            config_instance,
        )
        trainer.run()
    else:
        trainer = NeuralRingerTrainer(
            config_instance,
        )
        trainer.run()

    log.info("Fim da execução")


if __name__ == "__main__":
    args = parser.parse_args()
    log.info(args)
    main(args)
