
import logging
from pathlib import Path


LOG_FILE = Path(__file__).with_name("app.log")


def setup_logging():
    """配置应用程序的统一日志系统。"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(
                LOG_FILE,
                encoding="utf-8",
            ),
        ],
        force=True,
    )
