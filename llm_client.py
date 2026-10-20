
from openai import (
    APIConnectionError,
    APITimeoutError,
    AuthenticationError,
    InternalServerError,
    OpenAI,
    RateLimitError,
)

import config
import usage_tracker
from llm_result import LLMResult
import logging

logger = logging.getLogger(__name__)

config.validate_config()

client = OpenAI(
    api_key=config.API_KEY,
    base_url=config.BASE_URL,
    timeout=30.0,
    max_retries=1,
)


def stream_answer(messages):
    """发送对话请求，流式输出回答并返回结构化结果。"""
    answer_parts = []
    usage = None

    logger.info("开始调用模型")

    try:
        stream = client.chat.completions.create(
            model=config.MODEL,
            messages=messages,
            max_tokens=1000,
            stream=True,
            stream_options={"include_usage": True},
        )

        for chunk in stream:
            if chunk.usage is not None:
                usage = chunk.usage

            if not chunk.choices:
                continue

            content = chunk.choices[0].delta.content

            if content:
                print(content, end="", flush=True)
                answer_parts.append(content)

        print()
        answer = "".join(answer_parts)

        if usage is not None:
            usage_tracker.record_usage(usage)
        else:
            logger.warning("本次请求没有返回 Token 用量统计")
            print("本次请求没有返回 Token 用量统计。")

        return LLMResult(
            success=True,
            answer=answer,
        )

    except APITimeoutError as error:
        logger.exception("模型请求超时")
        print(f"\n模型请求超时：{error}")
        return LLMResult(success=False, error="timeout")

    except AuthenticationError as error:
        logger.exception("API 认证失败，请检查 API Key")
        print(f"\nAPI 认证失败，请检查 API Key：{error}")
        return LLMResult(success=False, error="authentication")

    except APIConnectionError as error:
        logger.exception("模型网络连接失败")
        print(f"\n模型网络连接失败：{error}")
        return LLMResult(success=False, error="connection")

    except RateLimitError as error:
        logger.exception("API 请求频率或额度限制")
        print(f"\nAPI 请求频率或额度限制：{error}")
        return LLMResult(success=False, error="rate_limit")

    except InternalServerError as error:
        logger.exception("API 服务端发生错误")
        print(f"\nAPI 服务端发生错误：{error}")
        return LLMResult(success=False, error="server_error")

    except Exception as error:
        logger.exception("模型调用失败")
        print(f"\n模型调用失败：{error}")
        return LLMResult(success=False, error="unknown")
