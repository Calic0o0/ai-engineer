
import pytest

from session_manager import (
    MAX_HISTORY_TURNS,
    SYSTEM_PROMPT,
    SessionManager,
)


def test_new_session_contains_system_prompt():
    manager = SessionManager()

    messages = manager.get_messages("session-a")

    assert messages == [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        }
    ]


def test_sessions_are_isolated():
    manager = SessionManager()

    manager.add_message("session-a", "user", "你好")
    manager.add_message("session-b", "user", "再见")

    messages_a = manager.get_messages("session-a")
    messages_b = manager.get_messages("session-b")

    assert messages_a[1]["content"] == "你好"
    assert messages_b[1]["content"] == "再见"
    assert len(messages_a) == 2
    assert len(messages_b) == 2


def test_create_existing_session_does_not_reset_history():
    manager = SessionManager()

    manager.add_message("session-a", "user", "保留这条消息")
    manager.create_session("session-a")

    messages = manager.get_messages("session-a")

    assert messages[-1] == {
        "role": "user",
        "content": "保留这条消息",
    }


def test_get_messages_returns_copy():
    manager = SessionManager()

    messages = manager.get_messages("session-a")
    messages.append({
        "role": "user",
        "content": "外部修改",
    })

    stored_messages = manager.get_messages("session-a")

    assert len(stored_messages) == 1


@pytest.mark.parametrize("role", ["system", "tool", "invalid"])
def test_add_message_rejects_invalid_role(role):
    manager = SessionManager()

    with pytest.raises(ValueError, match="role 必须是 user 或 assistant"):
        manager.add_message("session-a", role, "测试")


def test_trim_history_keeps_system_prompt_and_recent_five_turns():
    manager = SessionManager()
    session_id = "session-a"

    for number in range(1, 8):
        manager.add_message(
            session_id,
            "user",
            f"问题 {number}",
        )
        manager.add_message(
            session_id,
            "assistant",
            f"回答 {number}",
        )

    manager.trim_history(session_id)
    messages = manager.get_messages(session_id)

    assert messages[0] == {
        "role": "system",
        "content": SYSTEM_PROMPT,
    }
    assert len(messages) == 1 + MAX_HISTORY_TURNS * 2
    assert messages[1]["content"] == "问题 3"
    assert messages[-1]["content"] == "回答 7"


def test_trim_history_removes_incomplete_turn():
    manager = SessionManager()
    session_id = "session-a"

    manager.add_message(session_id, "user", "已完成的问题")
    manager.add_message(session_id, "assistant", "已完成的回答")
    manager.add_message(session_id, "user", "尚未回答的问题")

    manager.trim_history(session_id)
    messages = manager.get_messages(session_id)

    assert len(messages) == 3
    assert messages[1]["content"] == "已完成的问题"
    assert messages[2]["content"] == "已完成的回答"
