
from threading import RLock

SYSTEM_PROMPT = """
你是一位耐心、严谨的 Python 编程老师。
请结合完整的对话历史回答问题。
回答尽量控制在 200-400 个汉字左右，
优先解释核心概念并提供一个代码例子。
只有用户明确要求详细解释时才展开。
"""

MAX_HISTORY_TURNS = 5


class SessionManager:
    """管理多个相互隔离的对话会话。"""

    def __init__(self):
        self._sessions: dict[str, list[dict[str, str]]] = {}
        self._lock = RLock()

    def create_session(self, session_id: str) -> None:
        """创建会话；已存在时不覆盖已有历史。"""
        with self._lock:
            if session_id not in self._sessions:
                self._sessions[session_id] = [
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT,
                    }
                ]

    def get_messages(self, session_id: str) -> list[dict[str, str]]:
        """获取会话历史；不存在的会话自动创建。"""
        with self._lock:
            self.create_session(session_id)
            return [
                message.copy()
                for message in self._sessions[session_id]
            ]

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
    ) -> None:
        """向会话历史添加一条消息。"""
        if role not in {"user", "assistant"}:
            raise ValueError("role 必须是 user 或 assistant")

        with self._lock:
            self.create_session(session_id)
            self._sessions[session_id].append(
                {
                    "role": role,
                    "content": content,
                }
            )

    def trim_history(self, session_id: str) -> None:
        """保留系统提示词和最近 5 轮完整对话。"""
        with self._lock:
            self.create_session(session_id)
            messages = self._sessions[session_id]

            complete_turns = []
            pending_user = None

            for message in messages[1:]:
                if message["role"] == "user":
                    pending_user = message
                elif (
                    message["role"] == "assistant"
                    and pending_user is not None
                ):
                    complete_turns.append(
                        (pending_user, message)
                    )
                    pending_user = None

            recent_turns = complete_turns[-MAX_HISTORY_TURNS:]

            self._sessions[session_id] = [
                messages[0],
                *[
                    message
                    for turn in recent_turns
                    for message in turn
                ],
            ]
