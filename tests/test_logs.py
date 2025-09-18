import logging

import pytest

from water_quality.logs import setup_logging


def test_logging_level_behavior_level_1(caplog):
    logger = setup_logging(verbose=1)

    logger.critical("This is a Critical message")
    logger.error("This is an Error message")

    record_levelnames = [record.levelname for record in caplog.records]
    assert "CRITICAL" in record_levelnames
    assert "ERROR" not in record_levelnames


def test_logging_level_behavior_level_2(caplog):
    logger = setup_logging(verbose=2)
    logger.critical("This is a Critical message")
    logger.error("This is an Error message")
    logger.warning("This is a Warning message")

    record_levelnames = [record.levelname for record in caplog.records]
    assert "CRITICAL" in record_levelnames
    assert "ERROR" in record_levelnames
    assert "WARNING" not in record_levelnames


def test_logging_level_behavior_level_3(caplog):
    logger = setup_logging(verbose=3)

    logger.critical("This is a Critical message")
    logger.error("This is an Error message")
    logger.warning("This is a Warning message")
    logger.info("This is an Info message")

    record_levelnames = [record.levelname for record in caplog.records]
    assert "CRITICAL" in record_levelnames
    assert "ERROR" in record_levelnames
    assert "WARNING" in record_levelnames
    assert "INFO" not in record_levelnames


def test_logging_level_behavior_level_4(caplog):
    logger = setup_logging(verbose=4)
    logger.critical("This is a Critical message")
    logger.error("This is an Error message")
    logger.warning("This is a Warning message")
    logger.info("This is an Info message")
    logger.debug("This is a Debug message")

    record_levelnames = [record.levelname for record in caplog.records]
    assert "CRITICAL" in record_levelnames
    assert "ERROR" in record_levelnames
    assert "WARNING" in record_levelnames
    assert "INFO" in record_levelnames
    assert "DEBUG" not in record_levelnames


def test_logging_level_behavior_level_5(caplog):
    logger = setup_logging(verbose=5)
    logger.critical("This is a Critical message")
    logger.error("This is an Error message")
    logger.warning("This is a Warning message")
    logger.info("This is an Info message")
    logger.debug("This is a Debug message")

    record_levelnames = [record.levelname for record in caplog.records]
    assert "CRITICAL" in record_levelnames
    assert "ERROR" in record_levelnames
    assert "WARNING" in record_levelnames
    assert "INFO" in record_levelnames
    assert "DEBUG" in record_levelnames


def test_invalid_logging_level():
    with pytest.raises(ValueError):
        logger = setup_logging(verbose=6)
