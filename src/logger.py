import logging
import os
from config import Config


def get_logger(name: str):

    os.makedirs(Config.LOGS_PATH, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(Config.LOG_LEVEL)

    if not logger.handlers:
        file_handler = logging.FileHandler(
            os.path.join(Config.LOGS_PATH, "app.log")
        )
        formatter = logging.Formatter(Config.LOG_FORMAT)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger