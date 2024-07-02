import time
from .logging_config import setup_logging

logger = setup_logging()


def log_and_time_execution(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        logger.info(f"START: {func.__name__.replace('_', ' ').capitalize()}")
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = round((end_time - start_time) / 60, 2)
        logger.info(f"FINISH: {func.__name__.replace(
            '_', ' ').capitalize()} in {elapsed_time} min")
        return result
    return wrapper
