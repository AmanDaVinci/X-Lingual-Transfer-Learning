import logging
import logging.handlers

from pathlib import Path


def init_logging(log_path, log_level=logging.INFO):
    """Initialize logger"""
    log_path = Path('/tmp') / Path(log_path)
    logging.basicConfig(level=log_level) # set minimum log level
    logFormatter = logging.Formatter("%(asctime)s [%(levelname)-5.5s]  %(message)s")
    rootLogger = logging.getLogger()

    # FileHandler created with log file path.
    log_path.mkdir(exist_ok=True, parents=True)
    fileHandler = logging.handlers.WatchedFileHandler(f'{log_path}/logs.txt')
    fileHandler.setFormatter(logFormatter)
    rootLogger.addHandler(fileHandler)
