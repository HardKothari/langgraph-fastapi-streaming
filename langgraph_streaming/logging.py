# Global Imports
from loguru import logger

# Custom filter to exclude traceback. This will remove traceback so only use logger orders to ensure that traceback is available for initial loggers.
def exclude_traceback(record):
    if record["level"].name in {"ERROR", "EXCEPTION"}:
        # Remove the traceback if the log level is ERROR or EXCEPTION
        record["exception"] = None
        record["message"] = f"{record['message']}.\nCheck error.log for more detailed info."
    return True  # Always pass the record to be logged

# Adding error log first to include tracebacks
logger.add("logs/error.log",
           rotation="10 MB",
           level="ERROR",
           filter=lambda record: record["level"].name == "ERROR",
           format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {module}:{function}:{line} | {message}")

# General log file for all methods which would exclude tracebacks
logger.add("logs/general.log",
           rotation="10 MB",
           filter=exclude_traceback,
           #    level="INFO",
           colorize=True,
           format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {module}:{function}:{line} | {message}")

# Export the logger
__all__ = ['logger']