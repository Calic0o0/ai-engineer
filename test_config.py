
import pytest

import config


def test_validate_config_success(monkeypatch):
    monkeypatch.setattr(config, "API_KEY", "test-key")
    monkeypatch.setattr(
        config,
        "BASE_URL",
        "https://example.com/v1",
    )
    monkeypatch.setattr(config, "MODEL", "test-model")

    config.validate_config()


def test_validate_config_missing_api_key(monkeypatch):
    monkeypatch.setattr(config, "API_KEY", None)
    monkeypatch.setattr(
        config,
        "BASE_URL",
        "https://example.com/v1",
    )
    monkeypatch.setattr(config, "MODEL", "test-model")

    with pytest.raises(ValueError, match="SILICONFLOW_API_KEY"):
        config.validate_config()
