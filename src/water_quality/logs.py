import inspect
import logging
import sys


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """
    Setup logging to print to stdout with configurable verbosity.
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Suppress noisy credential discovery messages from botocore
    logging.getLogger("botocore.credentials").setLevel(logging.WARNING)

    if not root_logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        formatter = logging.Formatter(
            "%(asctime)s %(name)s [%(levelname)s]: %(message)s"
        )
        handler.setFormatter(formatter)
        root_logger.addHandler(handler)

    caller_module = inspect.stack()[1].frame.f_globals["__name__"]
    return logging.getLogger(caller_module)
