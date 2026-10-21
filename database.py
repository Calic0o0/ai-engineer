from sqlalchemy import URL, create_engine, text

import config


database_url = URL.create(
    drivername="mysql+mysqlconnector",
    username=config.DB_USER,
    password=config.DB_PASSWORD,
    host=config.DB_HOST,
    port=config.DB_PORT,
    database=config.DB_NAME,
    query={"charset": "utf8mb4"},
)

engine = create_engine(
    database_url,
    pool_pre_ping=True,
)


def check_database_connection() -> bool:
    """检查 MySQL 是否可以正常连接。"""
    with engine.connect() as connection:
        connection.execute(text("SELECT 1"))
    return True
