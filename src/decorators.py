import time
import psutil
import functools
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


def log_ram_usage(func):
    @functools.wraps(func)
    def wrapper_log_ram_usage(*args, **kwargs):
        process = psutil.Process()
        ram_usage_before = process.memory_info().rss / 1024 ** 2 / 1000  # Convert bytes to GB
        logger.info(f"{func.__name__} (before) - RAM usage: {ram_usage_before:.2f} GB")
        result = func(*args, **kwargs)
        ram_usage_after = process.memory_info().rss / 1024 ** 2 / 1000  # Convert bytes to GB
        logger.info(f"{func.__name__} (after) - RAM usage: {ram_usage_after:.2f} GB")
        return result
    return wrapper_log_ram_usage
