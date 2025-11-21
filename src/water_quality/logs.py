import inspect
import logging
import sys


def setup_logging(verbose: int = 4):
    """
    Setup logging to print to stdout with configurable verbosity.
    """
    if verbose == 1:
        level = logging.CRITICAL
    elif verbose == 2:
        level = logging.ERROR
    elif verbose == 3:
        level = logging.WARNING
    elif verbose == 4:
        level = logging.INFO
    elif verbose == 5:
        level = logging.DEBUG
    else:
        raise ValueError("Maximum verbosity is -vvvvv (verbose=5)")

    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    if not root_logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        formatter = logging.Formatter(
            "%(asctime)s %(name)s [%(levelname)s]: %(message)s"
        )
        handler.setFormatter(formatter)
        root_logger.addHandler(handler)

    caller_module = inspect.stack()[1].frame.f_globals["__name__"]
    return logging.getLogger(caller_module)
