
import main


def test_trim_history_keeps_system_prompt():
    main.messages[:] = [
        {"role": "system", "content": "系统提示词"},
        {"role": "user", "content": "问题 1"},
        {"role": "assistant", "content": "回答 1"},
    ]

    main.trim_history()

    assert main.messages[0] == {
        "role": "system",
        "content": "系统提示词",
    }


def test_trim_history_keeps_latest_five_turns():
    main.messages[:] = [
        {"role": "system", "content": "系统提示词"},
    ]

    for i in range(1, 8):
        main.messages.append({
            "role": "user",
            "content": f"问题 {i}",
        })
        main.messages.append({
            "role": "assistant",
            "content": f"回答 {i}",
        })

    main.trim_history()

    assert len(main.messages) == 11
    assert main.messages[1]["content"] == "问题 3"
    assert main.messages[-1]["content"] == "回答 7"


def test_trim_history_keeps_complete_turns_only():
    main.messages[:] = [
        {"role": "system", "content": "系统提示词"},
        {"role": "user", "content": "问题 1"},
        {"role": "assistant", "content": "回答 1"},
        {"role": "user", "content": "问题 2"},
    ]

    main.trim_history()

    assert len(main.messages) == 3
    assert main.messages[1]["content"] == "问题 1"
    assert main.messages[2]["content"] == "回答 1"


def test_ask_ai_removes_question_when_answer_is_empty(monkeypatch):
    main.messages[:] = [
        {"role": "system", "content": "系统提示词"},
    ]

    def fake_stream_answer(messages):
        return ""

    monkeypatch.setattr(
        main.llm_client,
        "stream_answer",
        fake_stream_answer,
    )

    main.ask_ai("测试问题")

    assert len(main.messages) == 1
    assert main.messages[0]["role"] == "system"


def test_ask_ai_saves_question_and_answer(monkeypatch):
    main.messages[:] = [
        {"role": "system", "content": "系统提示词"},
    ]

    def fake_stream_answer(messages):
        return "这是模拟的 AI 回答"

    monkeypatch.setattr(
        main.llm_client,
        "stream_answer",
        fake_stream_answer,
    )

    main.ask_ai("什么是 Python？")

    assert len(main.messages) == 3
    assert main.messages[0]["role"] == "system"
    assert main.messages[1] == {
        "role": "user",
        "content": "什么是 Python？",
    }
    assert main.messages[2] == {
        "role": "assistant",
        "content": "这是模拟的 AI 回答",
    }
