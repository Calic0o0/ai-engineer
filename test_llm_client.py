
import llm_client
from types import SimpleNamespace

import logging

import httpx
from openai import (
    APIConnectionError,
    APITimeoutError,
    AuthenticationError,
    InternalServerError,
    RateLimitError,
)

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
    caplog,
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
    assert "模型调用失败" in caplog.text

def test_stream_answer_handles_timeout(monkeypatch, capsys, caplog):
    """模拟 API 请求超时，验证异常处理逻辑。"""

    def fake_create(**kwargs):
        raise APITimeoutError(
            request=httpx.Request(
                "POST",
                "https://api.siliconflow.cn/v1/chat/completions",
            )
        )

    monkeypatch.setattr(
        llm_client.client.chat.completions,
        "create",
        fake_create,
    )

    answer = llm_client.stream_answer(
        [{"role": "user", "content": "测试超时"}]
    )

    captured = capsys.readouterr()

    assert answer is None
    assert "模型请求超时" in captured.out
    assert "模型请求超时" in caplog.text

def test_stream_answer_handles_connection_error(
    monkeypatch,
    capsys,
    caplog,
):
    """模拟网络连接失败，验证异常处理逻辑。"""

    def fake_create(**kwargs):
        raise APIConnectionError(
            request=httpx.Request(
                "POST",
                "https://api.siliconflow.cn/v1/chat/completions",
            )
        )

    monkeypatch.setattr(
        llm_client.client.chat.completions,
        "create",
        fake_create,
    )

    answer = llm_client.stream_answer(
        [{"role": "user", "content": "测试网络错误"}]
    )

    captured = capsys.readouterr()

    assert answer is None
    assert "模型网络连接失败" in captured.out
    assert "模型网络连接失败" in caplog.text

def test_stream_answer_handles_authentication_error(
    monkeypatch,
    capsys,
    caplog,
):
    """模拟 API 认证失败，验证异常处理逻辑。"""

    def fake_create(**kwargs):
        raise AuthenticationError(
            message="模拟认证失败",
            response=httpx.Response(
                401,
                request=httpx.Request(
                    "POST",
                    "https://api.siliconflow.cn/v1/chat/completions",
                ),
            ),
            body=None,
        )

    monkeypatch.setattr(
        llm_client.client.chat.completions,
        "create",
        fake_create,
    )

    answer = llm_client.stream_answer(
        [{"role": "user", "content": "测试认证错误"}]
    )

    captured = capsys.readouterr()

    assert answer is None
    assert "API 认证失败" in captured.out
    assert "API 认证失败" in caplog.text

def test_stream_answer_handles_rate_limit_error(
    monkeypatch,
    capsys,
    caplog,
):
    """模拟 API 请求频率或额度限制。"""

    def fake_create(**kwargs):
        raise RateLimitError(
            message="模拟请求频率限制",
            response=httpx.Response(
                429,
                request=httpx.Request(
                    "POST",
                    "https://api.siliconflow.cn/v1/chat/completions",
                ),
            ),
            body=None,
        )

    monkeypatch.setattr(
        llm_client.client.chat.completions,
        "create",
        fake_create,
    )

    answer = llm_client.stream_answer(
        [{"role": "user", "content": "测试请求频率限制"}]
    )

    captured = capsys.readouterr()

    assert answer is None
    assert "API 请求频率或额度限制" in captured.out
    assert "API 请求频率或额度限制" in caplog.text

def test_stream_answer_handles_internal_server_error(
    monkeypatch,
    capsys,
    caplog,
):
    """模拟 API 服务端发生错误。"""

    def fake_create(**kwargs):
        raise InternalServerError(
            message="模拟服务端错误",
            response=httpx.Response(
                500,
                request=httpx.Request(
                    "POST",
                    "https://api.siliconflow.cn/v1/chat/completions",
                ),
            ),
            body=None,
        )

    monkeypatch.setattr(
        llm_client.client.chat.completions,
        "create",
        fake_create,
    )

    answer = llm_client.stream_answer(
        [{"role": "user", "content": "测试服务端错误"}]
    )

    captured = capsys.readouterr()

    assert answer is None
    assert "API 服务端发生错误" in captured.out
    assert "API 服务端发生错误" in caplog.text

def test_stream_answer_joins_stream_chunks(monkeypatch, capsys, caplog):
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

    caplog.set_level(logging.INFO, logger="llm_client")

    answer = llm_client.stream_answer(messages)

    assert answer == "你好，世界！"
    assert "开始调用模型" in caplog.text

    capsys.readouterr()

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

def test_stream_answer_without_usage(monkeypatch, capsys, caplog):
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
    caplog.set_level(logging.WARNING, logger="llm_client")

    answer = llm_client.stream_answer(
        [{"role": "user", "content": "测试缺失用量"}]
    )

    captured = capsys.readouterr()

    assert answer == "正常回答"
    assert "本次请求没有返回 Token 用量统计" in captured.out
    assert "本次请求没有返回 Token 用量统计" in caplog.text

def test_client_has_one_retry():
    """确认模型客户端最多额外重试一次。"""
    assert llm_client.client.max_retries == 1

def test_sdk_retries_rate_limit_once():
    """验证 SDK 遇到 HTTP 429 时只额外重试一次。"""
    import httpx2
    from openai import OpenAI, RateLimitError

    request_count = 0

    def mock_handler(request):
        nonlocal request_count
        request_count += 1

        return httpx2.Response(
            status_code=429,
            json={
                "error": {
                    "message": "Rate limit test",
                    "type": "rate_limit_error",
                }
            },
        )

    transport = httpx2.MockTransport(mock_handler)
    http_client = httpx2.Client(transport=transport)

    client = OpenAI(
        api_key="test-key",
        base_url="https://example.com/v1",
        http_client=http_client,
        max_retries=1,
    )

    try:
        try:
            client.chat.completions.create(
                model="test-model",
                messages=[{"role": "user", "content": "test"}],
            )
        except RateLimitError:
            pass
        else:
            raise AssertionError("预期 SDK 抛出 RateLimitError")

        assert request_count == 2, (
            f"预期请求 2 次，实际请求 {request_count} 次"
        )
    finally:
        client.close()
        http_client.close()

if __name__ == "__main__":
    manual_test()