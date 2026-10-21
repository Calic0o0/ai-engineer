from unittest.mock import MagicMock

import pytest

import database


def test_check_database_connection_success(monkeypatch):
    engine_mock = MagicMock()
    connection_mock = engine_mock.connect.return_value.__enter__.return_value

    monkeypatch.setattr(database, "engine", engine_mock)

    result = database.check_database_connection()

    assert result is True
    engine_mock.connect.assert_called_once()
    connection_mock.execute.assert_called_once()


def test_check_database_connection_failure(monkeypatch):
    engine_mock = MagicMock()
    engine_mock.connect.side_effect = RuntimeError("database unavailable")

    monkeypatch.setattr(database, "engine", engine_mock)

    with pytest.raises(RuntimeError, match="database unavailable"):
        database.check_database_connection()


def test_session_local_configuration():
    assert database.SessionLocal.kw["autoflush"] is False
    assert database.SessionLocal.kw["expire_on_commit"] is False
    assert database.SessionLocal.kw["bind"] is database.engine


def test_get_db_session_closes_session(monkeypatch):
    session_mock = MagicMock()
    monkeypatch.setattr(
        database,
        "SessionLocal",
        MagicMock(return_value=session_mock),
    )

    with database.get_db_session() as session:
        assert session is session_mock

    session_mock.close.assert_called_once()


def test_get_db_session_closes_session_on_exception(monkeypatch):
    session_mock = MagicMock()
    monkeypatch.setattr(
        database,
        "SessionLocal",
        MagicMock(return_value=session_mock),
    )

    with pytest.raises(RuntimeError, match="test error"):
        with database.get_db_session():
            raise RuntimeError("test error")

    session_mock.close.assert_called_once()
