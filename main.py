import argparse
import logging
from logging.config import fileConfig
from src.core.Trainer import Trainer
import os
import numpy as np
from datetime import datetime


CONFIG_FILE = 'logging.ini'
LOG_DIR = 'log'
os.makedirs(LOG_DIR, exist_ok=True)
log_filename = f"{LOG_DIR}/execution_{datetime.now().strftime('%Y%m%d%H%M%S')}.log"

fileConfig(CONFIG_FILE, 
           defaults={'log_file_path': log_filename})
logger = logging.getLogger(__name__)

# parser = argparse.ArgumentParser(prog='TrainerRunner')
# parser.add_argument('percentage')   
# parser.add_argument('model_tag')   
# parser.add_argument('-fp', '--folder_path', default=None)
# parser.add_argument('--debug', default=False)
# args = parser.parse_args()


# trainer = Trainer(percentage = float(args.percentage),
#                           et_range = np.arange(0,8),
#                          eta_range = np.arange(0,9), 
#                          model_tag= str(args.model_tag),
#                          folder_path= args.folder_path)
# trainer.run()

trainer = Trainer(percentage = 1.0,
                  et_range = np.arange(0,1),
                  eta_range = np.arange(0,1), 
                  model_tag= "V1",
                  debug=True)
trainer.run()
logger.info("Fim da execução")