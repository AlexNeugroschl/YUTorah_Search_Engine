import logging
import os


def setup_logging(log_level=logging.INFO):
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=log_level, format=log_format)

    # Create a file handler if needed
    log_file = os.getenv('LOG_FILE', 'app.log')
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(log_level)
    file_handler.setFormatter(logging.Formatter(log_format))

    # Get the root logger and add the file handler to it
    logger = logging.getLogger()
    logger.addHandler(file_handler)

    # Optionally, prevent logging from propagating to the root logger
    logger.propagate = False

    return logger
