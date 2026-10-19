
import llm_client
from types import SimpleNamespace

def manual_test():
    """手动调用真实 API，检查模型是否能正常回答。"""
    messages = [
        {
            "role": "system",
            "content": "你是一个简洁的 Python 编程老师。",
        },
        {
            "role": "user",
            "content": "用一句话解释 Python 函数。",
        },
    ]

    answer = llm_client.stream_answer(messages)

    if answer:
        print("\n测试成功：已获取完整回答。")
    else:
        print("\n测试失败：没有获取到回答。")


def test_stream_answer_returns_none_when_api_fails(
    monkeypatch,
    capsys,
):
    """模拟 API 调用失败，验证异常处理逻辑。"""

    def fake_create(**kwargs):
        raise RuntimeError("模拟 API 故障")

    monkeypatch.setattr(
        llm_client.client.chat.completions,
        "create",
        fake_create,
    )

    messages = [
        {
            "role": "system",
            "content": "你是一个 Python 编程老师。",
        },
        {
            "role": "user",
            "content": "什么是 Python？",
        },
    ]

    answer = llm_client.stream_answer(messages)

    assert answer is None

    captured = capsys.readouterr()
    assert "模型调用失败" in captured.out
    assert "模拟 API 故障" in captured.out



def test_stream_answer_joins_stream_chunks(monkeypatch, capsys):
    chunks = [
        SimpleNamespace(
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(content="你好"),
                )
            ],
            usage=None,
        ),
        SimpleNamespace(
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(content="，"),
                )
            ],
            usage=None,
        ),
        SimpleNamespace(
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(content="世界！"),
                )
            ],
            usage=None,
        ),
    ]

    def fake_create(**kwargs):
        return iter(chunks)

    monkeypatch.setattr(
        llm_client.client.chat.completions,
        "create",
        fake_create,
    )

    messages = [
        {"role": "user", "content": "打个招呼"},
    ]

    answer = llm_client.stream_answer(messages)

    assert answer == "你好，世界！"
    capsys.readouterr()


if __name__ == "__main__":
    manual_test()


def test_stream_answer_records_token_usage(monkeypatch, capsys):
    usage = SimpleNamespace(
        prompt_tokens=10,
        completion_tokens=5,
        total_tokens=15,
    )

    chunks = [
        SimpleNamespace(
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(content="测试回答"),
                )
            ],
            usage=None,
        ),
        SimpleNamespace(
            choices=[],
            usage=usage,
        ),
    ]

    recorded_usages = []

    def fake_create(**kwargs):
        return iter(chunks)

    def fake_record_usage(received_usage):
        recorded_usages.append(received_usage)

    monkeypatch.setattr(
        llm_client.client.chat.completions,
        "create",
        fake_create,
    )
    monkeypatch.setattr(
        llm_client.usage_tracker,
        "record_usage",
        fake_record_usage,
    )

    answer = llm_client.stream_answer(
        [{"role": "user", "content": "测试 Token 记录"}]
    )

    assert answer == "测试回答"
    assert recorded_usages == [usage]
    capsys.readouterr()


def test_stream_answer_without_usage(monkeypatch, capsys):
    chunks = [
        SimpleNamespace(
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(content="正常回答"),
                )
            ],
            usage=None,
        ),
    ]

    def fake_create(**kwargs):
        return iter(chunks)

    def unexpected_record_usage(usage):
        raise AssertionError("没有 Token 用量时不应记录统计")

    monkeypatch.setattr(
        llm_client.client.chat.completions,
        "create",
        fake_create,
    )
    monkeypatch.setattr(
        llm_client.usage_tracker,
        "record_usage",
        unexpected_record_usage,
    )

    answer = llm_client.stream_answer(
        [{"role": "user", "content": "测试缺失用量"}]
    )

    captured = capsys.readouterr()

    assert answer == "正常回答"
    assert "本次请求没有返回 Token 用量统计" in captured.out
