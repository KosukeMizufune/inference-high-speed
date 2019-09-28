import logging
from logging import getLogger, FileHandler, Formatter
import os
from pathlib import Path

FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'


def get_logger(exp_version):
    if not os.path.exists('log'):
        os.mkdir('log')

    logger = getLogger(exp_version)
    logger.setLevel(logging.INFO)

    log_path = Path('log') / Path(exp_version + '.log')
    file_handler = FileHandler(log_path)
    file_handler.setLevel(logging.INFO)

    handler_format = Formatter(FORMAT)
    file_handler.setFormatter(handler_format)

    logger.addHandler(file_handler)
    return logger
