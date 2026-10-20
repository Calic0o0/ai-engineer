
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

import llm_client
from llm_result import LLMResult

app = FastAPI(
    title="AI Engineer API",
    description="我的 AI 后端服务",
    version="0.1.0",
)


class ChatRequest(BaseModel):
    message: str = Field(
        min_length=1,
        max_length=4000,
    )

    @field_validator("message")
    @classmethod
    def message_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("message 不能为空或只包含空格")
        return value


@app.get("/health")
def health_check():
    """检查 API 服务是否正常运行。"""
    return {
        "status": "ok",
        "message": "AI API is running",
    }


@app.post("/chat")
def chat(request: ChatRequest):
    """接收用户问题并返回大模型回答。"""
    messages = [
        {
            "role": "user",
            "content": request.message,
        }
    ]

    result = llm_client.stream_answer(messages)

    if not result.success:
        status_code = {
            "timeout": 504,
            "connection": 503,
            "rate_limit": 429,
            "authentication": 502,
            "server_error": 502,
            "unknown": 502,
        }.get(result.error, 502)

        raise HTTPException(
            status_code=status_code,
            detail={
                "success": False,
                "error": result.error,
            },
        )

    return {
        "success": True,
        "answer": result.answer,
    }
