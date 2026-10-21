from collections.abc import Iterator
from contextlib import contextmanager
from sqlalchemy import URL, create_engine, text
from sqlalchemy.orm import Session, sessionmaker

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


SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    expire_on_commit=False,
)


@contextmanager
def get_db_session() -> Iterator[Session]:
    """创建数据库 Session，并在使用后关闭。"""
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


def check_database_connection() -> bool:
    """检查 MySQL 是否可以正常连接。"""
    with engine.connect() as connection:
        connection.execute(text("SELECT 1"))
    return True
