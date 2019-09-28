from functools import wraps
import time

from local_lib.utils.logging import get_logger


def stop_watch(version):
    def _stop_watch(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            elapsed_time = time.time() - start_time
            get_logger(version).info(elapsed_time)
            return result
        return wrapper
    return _stop_watch
