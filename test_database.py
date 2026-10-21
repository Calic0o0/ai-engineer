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
