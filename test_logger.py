
import logging

import logger
import pytest

def test_setup_logging_configures_root_logger():
    logger.setup_logging()

    root_logger = logging.getLogger()

    assert root_logger.level == logging.INFO
    assert root_logger.handlers


def test_setup_logging_creates_log_file(tmp_path, monkeypatch):
    log_file = tmp_path / "app.log"

    monkeypatch.setattr(logger, "LOG_FILE", log_file)

    logger.setup_logging()

    test_logger = logging.getLogger("test_logger")
    test_logger.info("测试日志内容")

    for handler in logging.getLogger().handlers:
        handler.flush()

    assert log_file.exists()
    assert "测试日志内容" in log_file.read_text(encoding="utf-8")





@pytest.fixture(autouse=True)
def cleanup_logging_handlers():
    yield

    root_logger = logging.getLogger()

    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
        handler.close()
