
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator


import logging
import time
import uuid
import llm_client
from llm_result import LLMResult
from logger import setup_logging


setup_logging()
logger = logging.getLogger(__name__)


app = FastAPI(
    title="AI Engineer API",
    description="我的 AI 后端服务",
    version="0.1.0",
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id

    start_time = time.perf_counter()

    try:
        response = await call_next(request)
    except Exception:
        logger.exception(
            "request_failed | request_id=%s | method=%s | path=%s",
            request_id,
            request.method,
            request.url.path,
        )
        raise

    duration_ms = (time.perf_counter() - start_time) * 1000
    response.headers["X-Request-ID"] = request_id

    logger.info(
        "request_completed | request_id=%s | method=%s | path=%s | status_code=%s | duration_ms=%.2f",
        request_id,
        request.method,
        request.url.path,
        response.status_code,
        duration_ms,
    )

    return response


ERROR_STATUS_CODES = {
    "timeout": 504,
    "connection": 503,
    "rate_limit": 429,
    "authentication": 502,
    "server_error": 502,
    "unknown": 502,
    "invalid_request": 422,
}

ERROR_MESSAGES = {
    "timeout": "模型服务响应超时，请稍后重试",
    "connection": "暂时无法连接模型服务，请稍后重试",
    "rate_limit": "模型服务请求过于频繁，请稍后重试",
    "authentication": "模型服务认证失败，请联系管理员",
    "server_error": "模型服务暂时不可用，请稍后重试",
    "unknown": "模型请求失败，请稍后重试",
    "invalid_request": "请求参数无效，请检查输入内容",
}

@app.exception_handler(HTTPException)
async def http_exception_handler(
    request: Request,
    exc: HTTPException,
):
    error = exc.detail

    if isinstance(error, dict):
        error_code = error.get("error", "unknown")
    else:
        error_code = str(error)

    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": error_code,
            "message": ERROR_MESSAGES.get(
                error_code,
                "请求处理失败，请稍后重试",
            ),
        },
        headers=exc.headers,
)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError,
):
    return JSONResponse(
        status_code=422,
        content={
            "success": False,
            "error": "invalid_request",
            "message": ERROR_MESSAGES["invalid_request"],
        },
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


class ChatResponse(BaseModel):
    success: bool
    answer: str | None = None


class ErrorResponse(BaseModel):
    success: bool = False
    error: str


@app.get("/health")
def health_check():
    """检查 API 服务是否正常运行。"""
    return {
        "status": "ok",
        "message": "AI API is running",
    }


@app.post("/chat", response_model=ChatResponse)
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
        status_code = ERROR_STATUS_CODES.get(result.error, 502)

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
