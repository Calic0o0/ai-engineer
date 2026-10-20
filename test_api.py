
from fastapi.testclient import TestClient
from llm_result import LLMResult
from api import app, ERROR_MESSAGES, ERROR_STATUS_CODES
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
        "message": "模型服务响应超时，请稍后重试",
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
        "message": "请求参数无效，请检查输入内容",
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
        "message": "请求参数无效，请检查输入内容",
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
        "message": "请求参数无效，请检查输入内容",
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
    assert response.json()["message"] == ERROR_MESSAGES[error]


def test_error_status_codes_mapping():
    assert ERROR_STATUS_CODES == {
        "timeout": 504,
        "connection": 503,
        "rate_limit": 429,
        "authentication": 502,
        "server_error": 502,
        "unknown": 502,
        "invalid_request": 422,
    }


def test_error_messages_mapping():
    assert ERROR_MESSAGES == {
        "timeout": "模型服务响应超时，请稍后重试",
        "connection": "暂时无法连接模型服务，请稍后重试",
        "rate_limit": "模型服务请求过于频繁，请稍后重试",
        "authentication": "模型服务认证失败，请联系管理员",
        "server_error": "模型服务暂时不可用，请稍后重试",
        "unknown": "模型请求失败，请稍后重试",
        "invalid_request": "请求参数无效，请检查输入内容",
    }
