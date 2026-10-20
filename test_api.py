
from fastapi.testclient import TestClient
from llm_result import LLMResult
from api import app
import pytest

client = TestClient(app)


def test_health_check():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "message": "AI API is running",
    }


def test_chat_success(monkeypatch):
    def fake_stream_answer(messages):
        assert messages == [
            {
                "role": "user",
                "content": "你好",
            }
        ]
        return LLMResult(
            success=True,
            answer="你好！这是模拟的 AI 回答。",
        )

    monkeypatch.setattr(
        "api.llm_client.stream_answer",
        fake_stream_answer,
    )

    response = client.post(
        "/chat",
        json={"message": "你好"},
    )

    assert response.status_code == 200
    assert response.json() == {
        "success": True,
        "answer": "你好！这是模拟的 AI 回答。",
    }


def test_chat_failure(monkeypatch):
    def fake_stream_answer(messages):
        return LLMResult(
            success=False,
            error="timeout",
        )

    monkeypatch.setattr(
        "api.llm_client.stream_answer",
        fake_stream_answer,
    )

    response = client.post(
        "/chat",
        json={"message": "你好"},
    )

    assert response.status_code == 504
    assert response.json() == {
            "success": False,
            "error": "timeout",
    }


def test_chat_rejects_empty_message():
    response = client.post(
        "/chat",
        json={"message": ""},
    )

    assert response.status_code == 422
    assert response.json() == {
        "success": False,
        "error": "invalid_request",
    }


def test_chat_rejects_message_over_4000_characters():
    response = client.post(
        "/chat",
        json={"message": "a" * 4001},
    )

    assert response.status_code == 422
    assert response.json() == {
        "success": False,
        "error": "invalid_request",
    }


def test_chat_rejects_whitespace_only_message():
    response = client.post(
        "/chat",
        json={"message": "   \t  "},
    )

    assert response.status_code == 422
    assert response.json() == {
        "success": False,
        "error": "invalid_request",
    }



@pytest.mark.parametrize(
    ("error", "expected_status"),
    [
        ("connection", 503),
        ("rate_limit", 429),
        ("authentication", 502),
        ("server_error", 502),
        ("unknown", 502),
    ],
)
def test_chat_failure_status_codes(
    monkeypatch,
    error,
    expected_status,
):
    def fake_stream_answer(messages):
        return LLMResult(
            success=False,
            error=error,
        )

    monkeypatch.setattr(
        "api.llm_client.stream_answer",
        fake_stream_answer,
    )

    response = client.post(
        "/chat",
        json={"message": "你好"},
    )

    assert response.status_code == expected_status
    assert response.json()["error"] == error
