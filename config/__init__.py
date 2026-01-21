import os
import traceback

import toml
from loguru import logger

path = os.path.dirname(os.path.abspath(__file__))
try:
    settings = toml.load(path + "/config.toml")
except Exception as e:
    logger.bind(decorChaGee=True).warning(traceback.format_exc() + "--" + str(e))
    settings = toml.load(path + "/backup_config.toml")