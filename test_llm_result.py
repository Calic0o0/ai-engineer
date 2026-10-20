
from llm_result import LLMResult
import pytest

def test_llm_result_success():
    result = LLMResult(
        success=True,
        answer="你好",
        error=None,
    )

    assert result.success is True
    assert result.answer == "你好"
    assert result.error is None


def test_llm_result_failure():
    result = LLMResult(
        success=False,
        answer=None,
        error="timeout",
    )

    assert result.success is False
    assert result.answer is None
    assert result.error == "timeout"


def test_llm_result_defaults():
    result = LLMResult(success=True)

    assert result.answer is None
    assert result.error is None


def test_success_result_rejects_error():
    with pytest.raises(ValueError):
        LLMResult(
            success=True,
            answer="调用成功",
            error="timeout",
        )


def test_failure_result_requires_error():
    with pytest.raises(ValueError):
        LLMResult(
            success=False,
            answer=None,
            error=None,
        )
