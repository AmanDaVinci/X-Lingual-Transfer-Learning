import logging
from pathlib import Path

def init_logging(log_path=LOG_PATH, log_level=logging.DEBUG):
    """Initialize logger."""
    log_path = Path(log_path)
    logging.basicConfig(level=log_level) # set minimum log level
    logFormatter = logging.Formatter("%(asctime)s [%(levelname)-5.5s]  %(message)s")
    rootLogger = logging.getLogger()

    # FileHandler created with log file path.
    log_path.mkdir(exist_ok=True, parents=True)
    fileHandler = logging.FileHandler(f'{log_path}/logs.txt')
    fileHandler.setFormatter(logFormatter)
    rootLogger.addHandler(fileHandler)

    # This allows logs to be printed on console as well.
    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    rootLogger.addHandler(consoleHandler)