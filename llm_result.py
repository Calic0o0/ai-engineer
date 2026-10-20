
from dataclasses import dataclass

from dataclasses import dataclass

@dataclass
class LLMResult:
    """表示一次大语言模型调用的结果。"""

    success: bool
    answer: str | None = None
    error: str | None = None

    def __post_init__(self):
        """校验结果对象的状态是否一致。"""
        if self.success and self.error is not None:
            raise ValueError("成功结果不能包含错误信息")

        if not self.success and not self.error:
            raise ValueError("失败结果必须包含错误信息")
