
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


def test_main_exits_and_prints_usage_summary(monkeypatch, capsys):
    import main

    inputs = iter(["exit"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(inputs))

    summary_called = []
    monkeypatch.setattr(
        main.usage_tracker,
        "print_usage_summary",
        lambda: summary_called.append(True),
    )

    main.main()

    captured = capsys.readouterr()

    assert summary_called == [True]
    assert "AI" in captured.out
    assert "exit" in captured.out.lower()


def test_main_reprompts_when_input_is_empty(monkeypatch, capsys):
    import main

    inputs = iter(["", "exit"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(inputs))

    monkeypatch.setattr(
        main.usage_tracker,
        "print_usage_summary",
        lambda: None,
    )

    def unexpected_ask_ai(question):
        raise AssertionError("空输入不应该调用 ask_ai")

    monkeypatch.setattr(main, "ask_ai", unexpected_ask_ai)

    main.main()

    captured = capsys.readouterr()

    assert "exit" in captured.out.lower()
    assert "问题" in captured.out or "输入" in captured.out


def test_main_handles_keyboard_interrupt(monkeypatch, capsys):
    import main

    def raise_keyboard_interrupt(prompt=""):
        raise KeyboardInterrupt

    monkeypatch.setattr(
        "builtins.input",
        raise_keyboard_interrupt,
    )

    summary_called = []
    monkeypatch.setattr(
        main.usage_tracker,
        "print_usage_summary",
        lambda: summary_called.append(True),
    )

    main.main()

    captured = capsys.readouterr()

    assert "KeyboardInterrupt" not in captured.out
    assert summary_called == [True]
    assert "AI" in captured.out


def test_main_handles_eof_error(monkeypatch, capsys):
    import main

    def raise_eof_error(prompt=""):
        raise EOFError

    monkeypatch.setattr(
        "builtins.input",
        raise_eof_error,
    )

    summary_called = []
    monkeypatch.setattr(
        main.usage_tracker,
        "print_usage_summary",
        lambda: summary_called.append(True),
    )

    main.main()

    captured = capsys.readouterr()

    assert "收到退出信号" in captured.out
    assert summary_called == [True]


def test_main_processes_valid_question(monkeypatch, capsys):
    import main

    inputs = iter(["什么是 Python？", "exit"])
    monkeypatch.setattr(
        "builtins.input",
        lambda prompt="": next(inputs),
    )

    asked_questions = []
    monkeypatch.setattr(
        main,
        "ask_ai",
        lambda question: asked_questions.append(question),
    )
    monkeypatch.setattr(
        main.usage_tracker,
        "print_usage_summary",
        lambda: None,
    )

    main.main()

    captured = capsys.readouterr()

    assert asked_questions == ["什么是 Python？"]
    assert "AI" in captured.out
