import os
from dotenv import load_dotenv

# 加载项目中的 .env 文件
load_dotenv()

# 读取大模型 API 配置
API_KEY = os.getenv("SILICONFLOW_API_KEY")
BASE_URL = os.getenv("SILICONFLOW_BASE_URL")
MODEL = os.getenv("SILICONFLOW_MODEL")

# 读取 MySQL 数据库配置
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = int(os.getenv("DB_PORT", "3306"))
DB_NAME = os.getenv("DB_NAME", "ai_engineer")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")


def validate_config():
    """检查必要的 API 配置是否存在。"""
    missing = []

    if not API_KEY:
        missing.append("SILICONFLOW_API_KEY")

    if not BASE_URL:
        missing.append("SILICONFLOW_BASE_URL")

    if not MODEL:
        missing.append("SILICONFLOW_MODEL")

    if missing:
        raise ValueError(
            f".env 中缺少必要配置：{', '.join(missing)}"
        )
