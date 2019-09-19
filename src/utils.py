from functools import wraps
import logging
from logging import getLogger, FileHandler, Formatter
import time


def time_watch(func):
    logger = getLogger(__name__)
    logger.setLevel(logging.INFO)
    file_handler = FileHandler('log/inference.log')
    handler_format = Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(handler_format)
    logger.addHandler(file_handler)

    @wraps(func)
    def wrapper(*args):
        start_time = time.time()
        result = func(*args)
        elapsed_itme = time.time() - start_time
        logger.info(elapsed_itme)
        return result

    return wrapper
