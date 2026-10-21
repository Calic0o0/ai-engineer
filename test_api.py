
from fastapi.testclient import TestClient
from llm_result import LLMResult
from api import app, ERROR_MESSAGES, ERROR_STATUS_CODES
import pytest

client = TestClient(app)


def test_request_id_header():
    response = client.get("/health")

    request_id = response.headers.get("X-Request-ID")

    assert response.status_code == 200
    assert request_id is not None
    assert len(request_id) == 36


def test_request_ids_are_unique():
    first_response = client.get("/health")
    second_response = client.get("/health")

    first_request_id = first_response.headers["X-Request-ID"]
    second_request_id = second_response.headers["X-Request-ID"]

    assert first_request_id != second_request_id


def test_request_logging(caplog):
    with caplog.at_level("INFO", logger="api"):
        response = client.get("/health")

    assert response.status_code == 200
    assert "request_completed" in caplog.text
    assert "request_id=" in caplog.text
    assert "method=GET" in caplog.text
    assert "path=/health" in caplog.text
    assert "status_code=200" in caplog.text
    assert "duration_ms=" in caplog.text


def test_request_logging_for_404(caplog):
    with caplog.at_level("INFO", logger="api"):
        response = client.get("/not-found")

    assert response.status_code == 404
    assert "request_completed" in caplog.text
    assert "path=/not-found" in caplog.text
    assert "status_code=404" in caplog.text


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


def test_unexpected_exception_does_not_leak_details(monkeypatch):
    secret_detail = "SECRET_INTERNAL_DETAIL"

    def fake_stream_answer(messages):
        raise RuntimeError(secret_detail)

    monkeypatch.setattr(
        "api.llm_client.stream_answer",
        fake_stream_answer,
    )

    with TestClient(app, raise_server_exceptions=False) as test_client:
        response = test_client.post(
            "/chat",
            json={"message": "你好"},
        )

    assert response.status_code == 500
    assert secret_detail not in response.text
    assert "Traceback" not in response.text


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"message": 123},
        {"message": None},
        {"message": ["你好"]},
    ],
)
def test_chat_rejects_missing_or_invalid_message(payload):
    response = client.post("/chat", json=payload)

    assert response.status_code == 422
    assert response.json() == {
        "success": False,
        "error": "invalid_request",
        "message": "请求参数无效，请检查输入内容",
    }


def test_chat_accepts_message_with_exactly_4000_characters(monkeypatch):
    message = "a" * 4000

    def fake_stream_answer(messages):
        assert messages == [
            {
                "role": "user",
                "content": message,
            }
        ]
        return LLMResult(
            success=True,
            answer="测试回答",
        )

    monkeypatch.setattr(
        "api.llm_client.stream_answer",
        fake_stream_answer,
    )

    response = client.post(
        "/chat",
        json={"message": message},
    )

    assert response.status_code == 200
    assert response.json() == {
        "success": True,
        "answer": "测试回答",
    }


def test_chat_rejects_malformed_json():
    response = client.post(
        "/chat",
        content="{invalid json",
        headers={"Content-Type": "application/json"},
    )

    assert response.status_code == 422
    assert response.json() == {
        "success": False,
        "error": "invalid_request",
        "message": "请求参数无效，请检查输入内容",
    }


def test_chat_continues_existing_session(monkeypatch):
    from api import session_manager

    session_id = "test-continuation"
    session_manager._sessions.pop(session_id, None)
    received_messages = []

    def fake_stream_answer(messages):
        received_messages.append([message.copy() for message in messages])
        return LLMResult(
            success=True,
            answer=f"回答 {len(received_messages)}",
        )

    monkeypatch.setattr(
        "api.llm_client.stream_answer",
        fake_stream_answer,
    )

    first_response = client.post(
        "/chat",
        json={
            "session_id": session_id,
            "message": "什么是 Python 列表？",
        },
    )
    second_response = client.post(
        "/chat",
        json={
            "session_id": session_id,
            "message": "请给我一个例子",
        },
    )

    assert first_response.status_code == 200
    assert second_response.status_code == 200

    assert received_messages[1] == [
        received_messages[0][0],
        {
            "role": "user",
            "content": "什么是 Python 列表？",
        },
        {
            "role": "assistant",
            "content": "回答 1",
        },
        {
            "role": "user",
            "content": "请给我一个例子",
        },
    ]


def test_chat_sessions_are_isolated(monkeypatch):
    from api import session_manager

    session_manager._sessions.pop("test-session-a", None)
    session_manager._sessions.pop("test-session-b", None)
    received_messages = []

    def fake_stream_answer(messages):
        received_messages.append([message.copy() for message in messages])
        return LLMResult(success=True, answer="模拟回答")

    monkeypatch.setattr(
        "api.llm_client.stream_answer",
        fake_stream_answer,
    )

    client.post(
        "/chat",
        json={"session_id": "test-session-a", "message": "A 的问题"},
    )
    client.post(
        "/chat",
        json={"session_id": "test-session-b", "message": "B 的问题"},
    )

    assert received_messages[0][-1]["content"] == "A 的问题"
    assert len(received_messages[0]) == 2
    assert received_messages[1][-1]["content"] == "B 的问题"
    assert len(received_messages[1]) == 2


def test_chat_failure_does_not_save_session_turn(monkeypatch):
    from api import session_manager

    session_id = "test-failed-session"
    session_manager._sessions.pop(session_id, None)

    def fake_stream_answer(messages):
        return LLMResult(success=False, error="timeout")

    monkeypatch.setattr(
        "api.llm_client.stream_answer",
        fake_stream_answer,
    )

    response = client.post(
        "/chat",
        json={"session_id": session_id, "message": "失败的问题"},
    )

    assert response.status_code == 504
    assert session_manager.get_messages(session_id) == [
        {
            "role": "system",
            "content": session_manager.get_messages(session_id)[0]["content"],
        }
    ]


def test_chat_rejects_blank_session_id():
    response = client.post(
        "/chat",
        json={"session_id": "   ", "message": "你好"},
    )

    assert response.status_code == 422
    assert response.json()["error"] == "invalid_request"


def test_chat_without_session_id_remains_stateless(monkeypatch):
    received_messages = []

    def fake_stream_answer(messages):
        received_messages.append([message.copy() for message in messages])
        return LLMResult(success=True, answer="模拟回答")

    monkeypatch.setattr(
        "api.llm_client.stream_answer",
        fake_stream_answer,
    )

    for message in ("第一个问题", "第二个问题"):
        response = client.post("/chat", json={"message": message})
        assert response.status_code == 200

    assert received_messages == [
        [{"role": "user", "content": "第一个问题"}],
        [{"role": "user", "content": "第二个问题"}],
    ]
