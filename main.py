import argparse
import logging
import os
from datetime import datetime
from logging.config import fileConfig

import numpy as np

from src.core.Trainer import Trainer

CONFIG_FILE = "logging.ini"
LOG_DIR = "log"
os.makedirs(LOG_DIR, exist_ok=True)
log_filename = f"{LOG_DIR}/TrainerRunner_{datetime.now().strftime('%Y%m%d%H%M%S')}.log"

fileConfig(CONFIG_FILE, defaults={"log_file_path": log_filename})
log = logging.getLogger(__name__)

parser = argparse.ArgumentParser(prog="TrainerRunner")
parser.add_argument("--percentage", default=None, type=float)
parser.add_argument("--model_tag", default=None, type=str)
parser.add_argument("-fp", "--folder_path", default=None)
parser.add_argument("--debug", action="store_true")


def main(args: argparse.Namespace):
    if bool(args.debug):
        trainer = Trainer(
            percentage=1,
            et_range=np.arange(0, 1),
            eta_range=np.arange(0, 1),
            model_tag="V1",
            debug=args.debug,
        )
        trainer.run()
    else:
        trainer = Trainer(
            percentage=float(args.percentage),
            et_range=np.arange(0, 8),
            eta_range=np.arange(0, 1),
            model_tag=str(args.model_tag),
            folder_path=args.folder_path,
        )
        trainer.run()

    log.info("Fim da execução")


if __name__ == "__main__":
    args = parser.parse_args()
    log.info(args)
    main(args)
