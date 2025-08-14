# -*- coding: utf-8 -*-
"""
优化后的 Gemini API Python 代理，具有强大的流式重试和标准化错误响应功能。

此版本包含多项改进：
- 使用 Pydantic 进行稳健的配置管理。
- 使用 FastAPI 的依赖注入来管理 HTTPX 客户端。
- 重构了流处理逻辑以提高可读性。
- 为错误响应和请求体定义了 Pydantic 模型。
- 使用中间件为日志添加唯一的请求 ID，以便于追踪。

要运行此脚本：
1. 安装必要的库：
   pip install "fastapi[all]" httpx pydantic pydantic-settings python-dotenv
2. 创建一个 .env 文件用于你的配置。
3. 运行服务器：
   uvicorn your_script_name:app --host 0.0.0.0 --port 20000 --reload
"""
import asyncio
import contextvars
import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import AsyncGenerator, List, Set

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from pydantic_settings import BaseSettings

# --- 上下文变量，用于请求 ID ---
# 在异步应用中，使用 contextvars 是处理上下文局部状态的现代且安全的方式。
request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="N/A")

# --- 常量 ---
SSE_DATA_PREFIX = "data: "
SSE_EVENT_PREFIX = "event: "
ROLE_USER = "user"
ROLE_MODEL = "model"
NON_RETRYABLE_STATUSES: Set[int] = {400, 401, 403, 404, 429}
SSE_ENCODER = str.encode


# --- 使用 Pydantic 进行配置 ---
class Settings(BaseSettings):
    upstream_url_base: str = "https://generativelanguage.googleapis.com"
    max_consecutive_retries: int = 5
    debug_mode: bool = True
    retry_delay_ms: int = 1000
    swallow_thoughts_after_retry: bool = True
    request_timeout_seconds: int = 300

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# 加载配置
load_dotenv()
settings = Settings()


# --- 日志设置 ---
# 自定义格式化器，用于从上下文变量中向每条日志记录添加 request_id。
class RequestIdFormatter(logging.Formatter):
    def format(self, record):
        # 从上下文变量获取 request_id，如果未设置则默认为 "N/A"。
        record.request_id = request_id_var.get()
        return super().format(record)


# 配置根日志记录器以使用我们的自定义格式化器。
# 这种方式比 basicConfig 更健壮，尤其是在使用重载器时。
handler = logging.StreamHandler()
# 格式字符串现在包含了我们添加的 'request_id' 字段。
formatter = RequestIdFormatter(
    fmt="[%(levelname)s %(asctime)s] [%(request_id)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
handler.setFormatter(formatter)

# 获取根日志记录器，清除任何现有的处理器，并添加我们的新处理器。
root_logger = logging.getLogger()
if root_logger.hasHandlers():
    root_logger.handlers.clear()
root_logger.addHandler(handler)
root_logger.setLevel(logging.DEBUG if settings.debug_mode else logging.INFO)

# 同时配置 uvicorn 的日志记录器以使用我们的处理器，以正确捕获其日志。
# 我们将禁用其日志传播(propagate=False)，以防止日志事件被传递到根日志记录器，
# 从而避免根日志记录器再次处理相同的事件，导致日志重复输出。
uvicorn_error_logger = logging.getLogger("uvicorn.error")
uvicorn_error_logger.handlers = [handler]
uvicorn_error_logger.propagate = False

uvicorn_logger = logging.getLogger("uvicorn")
uvicorn_logger.handlers = [handler]
uvicorn_logger.propagate = False

logger = logging.getLogger(__name__)


# --- Pydantic 数据结构模型 ---
class GeminiPart(BaseModel):
    text: str


class GeminiContent(BaseModel):
    role: str
    parts: List[GeminiPart]


class GeminiRequest(BaseModel):
    contents: List[GeminiContent]

    class Config:
        extra = "allow"


# --- HTTP 客户端管理 ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """管理应用的生命周期，包括 HTTPX 客户端。"""
    global async_client
    logger.info("--- 正在初始化应用 ---")
    logger.info("--- 已加载配置 ---")
    logger.info(json.dumps(settings.model_dump(), indent=4, ensure_ascii=False))
    logger.info("--------------------------")

    async_client = httpx.AsyncClient(timeout=settings.request_timeout_seconds)
    logger.info("HTTPX 异步客户端已启动。")
    yield
    await async_client.aclose()
    logger.info("HTTPX 异步客户端已关闭。")
    logger.info("--- 应用关闭完成 ---")


async_client: httpx.AsyncClient
app = FastAPI(lifespan=lifespan)


# --- 中间件 ---
@app.middleware("http")
async def add_request_id_middleware(request: Request, call_next):
    """为每个传入的请求分配一个唯一的 ID 用于追踪。"""
    # 在此请求的上下文变量中设置请求 ID。
    token = request_id_var.set(str(uuid.uuid4()))

    start_time = time.time()
    logger.info(f"请求开始: {request.method} {request.url}")

    response = await call_next(request)

    process_time = (time.time() - start_time) * 1000
    response.headers["X-Request-ID"] = request_id_var.get()
    response.headers["X-Process-Time-Ms"] = str(process_time)
    logger.info(f"请求在 {process_time:.2f}ms 内完成. 状态码: {response.status_code}")

    # 请求完成后重置上下文变量。
    request_id_var.reset(token)
    return response


# CORS 中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- 辅助函数 ---
def status_to_google_status(code: int) -> str:
    """将 HTTP 状态码转换为 Google 的 RPC 状态字符串。"""
    status_map = {
        400: "INVALID_ARGUMENT", 401: "UNAUTHENTICATED", 403: "PERMISSION_DENIED",
        404: "NOT_FOUND", 429: "RESOURCE_EXHAUSTED", 500: "INTERNAL",
        503: "UNAVAILABLE", 504: "DEADLINE_EXCEEDED",
    }
    return status_map.get(code, "UNKNOWN")


async def standardize_error_response(upstream_resp: httpx.Response) -> JSONResponse:
    """从上游错误创建标准化的 JSON 错误响应。"""
    await upstream_resp.aread()
    upstream_text = upstream_resp.text
    logger.error(f"上游错误 ({upstream_resp.status_code}): {upstream_text}")

    standardized_error = {
        "error": {
            "code": upstream_resp.status_code,
            "message": upstream_resp.reason_phrase or "请求失败",
            "status": status_to_google_status(upstream_resp.status_code),
            "details": [{"@type": "proxy.upstream", "error_body": upstream_text[:20000]}]
        }
    }

    try:
        data = upstream_resp.json()
        if "error" in data and isinstance(data["error"], dict):
            # Overwrite with more specific details from the upstream error if available
            error_data = data["error"]
            standardized_error["error"]["message"] = error_data.get("message", standardized_error["error"]["message"])
            standardized_error["error"]["status"] = error_data.get("status", standardized_error["error"]["status"])
            # Pass through details directly without validation
            standardized_error["error"]["details"] = error_data.get("details", standardized_error["error"]["details"])
    except (json.JSONDecodeError, KeyError):
        logger.warning("无法从上游响应中解析出详细的错误结构。")
        pass

    return JSONResponse(
        content=standardized_error,
        status_code=upstream_resp.status_code
    )


async def sse_line_iterator(response: httpx.Response) -> AsyncGenerator[str, None]:
    """从 SSE 流中逐行生成数据，处理潜在的解码问题。"""
    buffer = ""
    async for chunk in response.aiter_bytes():
        buffer += chunk.decode("utf-8", errors="ignore")
        while '\n' in buffer:
            line, buffer = buffer.split('\n', 1)
            line = line.strip()
            if line:
                yield line
    if buffer.strip():
        yield buffer.strip()


def extract_sse_data(line: str) -> dict:
    """从 SSE 'data:' 行中提取 JSON 数据。"""
    if not line.startswith(SSE_DATA_PREFIX):
        return {}
    try:
        return json.loads(line[len(SSE_DATA_PREFIX):])
    except json.JSONDecodeError:
        logger.warning(f"从 SSE 行解码 JSON 失败: {line}")
        return {}


def build_retry_request_body(original_body: dict, accumulated_text: str) -> dict:
    """为重试尝试构建新的请求体。"""
    logger.debug(f"正在构建重试请求体。累积文本长度: {len(accumulated_text)}")
    retry_body = original_body.copy()

    history = []
    if accumulated_text.strip():
        history.append({"role": ROLE_MODEL, "parts": [{"text": accumulated_text}]})

    history.append({
        "role": ROLE_USER,
        "parts": [{"text": "Continue exactly where you left off without any preamble or repetition."}]
    })

    retry_body["contents"] = retry_body.get("contents", []) + history

    return retry_body


async def process_stream_with_retries(
        initial_response: httpx.Response,
        original_request: GeminiRequest,
        upstream_url: str,
        original_headers: dict
) -> AsyncGenerator[bytes, None]:
    """使用强大的重试机制处理 SSE 流。"""
    accumulated_text = ""
    retry_count = 0
    current_response = initial_response
    is_outputting_formal_text = False
    swallow_thoughts = False

    # 捕获并保留原始的 responseId，以确保在重试后对客户端保持一致。
    original_response_id = None

    while retry_count <= settings.max_consecutive_retries:
        interruption_reason = "UNKNOWN"
        clean_exit = False
        line_count = 0
        finish_reason = None

        try:
            async for line in sse_line_iterator(current_response):
                if settings.debug_mode:
                    line_count += 1
                    logger.debug(f"从上游接收到响应内容: SSE Line {line_count} {line}")

                data = extract_sse_data(line)
                if not data:
                    yield SSE_ENCODER(line + "\n\n")
                    continue

                # 确保整个流的 responseId 保持一致
                if original_response_id is None and "responseId" in data:
                    original_response_id = data["responseId"]
                    logger.debug(f"捕获到原始 responseId: {original_response_id}")

                if original_response_id is not None:
                    data["responseId"] = original_response_id

                # 更安全地解析响应，以处理空内容的情况
                candidate = data.get("candidates", [{}])[0]
                content = candidate.get("content", {})
                parts = content.get("parts", [])

                text_chunk = ""
                is_thought = False

                if parts:  # 仅当 parts 存在且非空时处理
                    content_part = parts[0]
                    text_chunk = content_part.get("text", "")
                    is_thought = content_part.get("thought") is True

                is_empty_content = not parts
                finish_reason = candidate.get("finishReason")

                # 在将数据发送给客户端之前，重新序列化可能已修改的 data 字典
                modified_line = f"{SSE_DATA_PREFIX}{json.dumps(data, ensure_ascii=False)}"
                yield SSE_ENCODER(modified_line + "\n\n")

                if swallow_thoughts:
                    if is_thought:
                        logger.debug("重试后吞掉'thought'块。")
                        if finish_reason:
                            interruption_reason = "FINISH_DURING_THOUGHT_SWALLOW"
                            break
                        continue
                    else:
                        logger.info("吞咽后收到第一个正式文本块。恢复正常流。")
                        swallow_thoughts = False

                if text_chunk and not is_thought:
                    accumulated_text += text_chunk
                    is_outputting_formal_text = True

                if finish_reason:
                    is_clean_stop_reason = finish_reason in ("STOP", "MAX_TOKENS")

                    # 增加对过早停止和空内容停止的检查
                    stopped_prematurely_on_thought = (
                            is_clean_stop_reason and is_thought and not is_outputting_formal_text
                    )

                    stopped_on_empty_content = is_clean_stop_reason and is_empty_content

                    if is_clean_stop_reason and not stopped_prematurely_on_thought and not stopped_on_empty_content:
                        logger.info(f"数据流以原因 '{finish_reason}' 正常结束。")
                        clean_exit = True
                    else:
                        if stopped_prematurely_on_thought:
                            logger.error(
                                f"数据流在输出任何正式内容前，在'thought'块后过早停止。原因: '{finish_reason}'。触发重试。"
                            )
                            interruption_reason = "PREMATURE_STOP_ON_THOUGHT"
                        elif stopped_on_empty_content:
                            logger.error(
                                f"数据流以空的 content 块停止。原因: '{finish_reason}'。触发重试。"
                            )
                            interruption_reason = "STOP_ON_EMPTY_CONTENT"
                        else:
                            logger.error(f"异常的结束原因 '{finish_reason}'。触发重试。")
                            interruption_reason = f"FINISH_{finish_reason}"
                    break

            if not clean_exit and not finish_reason:
                logger.error("数据流在没有结束原因的情况下突然终止 (DROP)。")
                interruption_reason = "DROP"

        except httpx.RequestError as e:
            logger.error(f"处理数据流时发生 HTTPX 错误: {e}", exc_info=True)
            interruption_reason = "FETCH_ERROR"
        finally:
            await current_response.aclose()

        if clean_exit:
            logger.info("=== 数据流成功完成 ===")
            return

        logger.error(
            f"数据流中断。原因: {interruption_reason}。重试 {retry_count + 1}/{settings.max_consecutive_retries}")

        retry_count += 1
        if retry_count > settings.max_consecutive_retries:
            break

        try:
            yield SSE_ENCODER(": heartbeat\n\n")
            logger.debug("发送 SSE 心跳以防止连接中断。")
        except GeneratorExit:
            logger.warning("在发送心跳时检测到客户端断开连接。正在中止重试。")
            return

        if settings.swallow_thoughts_after_retry and is_outputting_formal_text:
            swallow_thoughts = True

        await asyncio.sleep(settings.retry_delay_ms / 1000)

        try:
            retry_body = build_retry_request_body(original_request.model_dump(), accumulated_text)
            req = async_client.build_request("POST", upstream_url, headers=original_headers, json=retry_body)
            retry_response = await async_client.send(req, stream=True)

            if retry_response.status_code in NON_RETRYABLE_STATUSES:
                logger.error(f"重试时遇到致命的不可重试状态码 {retry_response.status_code}。")
                error_resp = await standardize_error_response(retry_response)
                error_sse_event = f"{SSE_EVENT_PREFIX}error\n{SSE_DATA_PREFIX}{error_resp.body.decode('utf-8')}\n\n"
                yield SSE_ENCODER(error_sse_event)
                return

            retry_response.raise_for_status()
            logger.info("重试成功，已重新建立数据流。")
            current_response = retry_response

        except httpx.HTTPError as e:
            logger.error(f"重试尝试失败: {e}", exc_info=True)

    logger.error("已超出最大重试次数限制。终止数据流。")
    error_payload = {
        "error": {
            "code": 504, "status": "DEADLINE_EXCEEDED",
            "message": f"代理重试次数限制 ({settings.max_consecutive_retries}) 已超出。最后原因: {interruption_reason}."
        }
    }
    yield SSE_ENCODER(f"{SSE_EVENT_PREFIX}error\n{SSE_DATA_PREFIX}{json.dumps(error_payload, ensure_ascii=False)}\n\n")


# --- API 路由 ---
@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def proxy_request(request: Request, path: str):
    """主代理端点，处理流式和非流式请求。"""
    upstream_url = f"{settings.upstream_url_base}/{path}"
    if request.query_params:
        upstream_url += f"?{request.query_params}"

    headers_to_forward = {
        k: v for k, v in request.headers.items()
        if k.lower() in ["authorization", "x-goog-api-key", "content-type"]
    }

    is_stream = "stream" in path or "alt=sse" in str(request.query_params)

    if is_stream and request.method == "POST":
        try:
            body_bytes = await request.body()
            if settings.debug_mode:
                logger.debug(f"接收到用户的流式请求体 (前100字符): {body_bytes.decode('utf-8')[:100]}")
            original_request_body = GeminiRequest.model_validate_json(body_bytes)
        except Exception as e:
            logger.error(f"流式请求的请求体无效: {e}", exc_info=True)
            return JSONResponse(status_code=400, content={"error": "无效的 JSON 请求体。"})

        try:
            req = async_client.build_request(
                "POST", upstream_url, headers=headers_to_forward,
                json=original_request_body.model_dump(exclude_unset=True)
            )
            initial_response = await async_client.send(req, stream=True)

            if not initial_response.is_success:
                return await standardize_error_response(initial_response)

            return StreamingResponse(
                process_stream_with_retries(
                    initial_response, original_request_body, upstream_url, headers_to_forward
                ),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
            )
        except httpx.RequestError as e:
            logger.error(f"连接上游服务器进行流式传输失败: {e}", exc_info=True)
            return JSONResponse(status_code=502, content={"error": "错误的网关"})

    else:  # 非流式请求
        try:
            body_bytes = await request.body()
            if settings.debug_mode:
                logger.debug(f"接收到非流式请求体: {body_bytes.decode('utf-8')[:100]}")

            req = async_client.build_request(
                request.method, upstream_url, headers=headers_to_forward, content=body_bytes
            )
            upstream_resp = await async_client.send(req)

            if settings.debug_mode:
                logger.debug(f"从上游接收到非流式响应内容: {upstream_resp.text}")

            if not upstream_resp.is_success:
                return await standardize_error_response(upstream_resp)

            response_headers = dict(upstream_resp.headers)
            response_headers.pop("content-encoding", None)
            response_headers.pop("transfer-encoding", None)

            return Response(
                content=upstream_resp.content,
                status_code=upstream_resp.status_code,
                headers=response_headers,
                media_type=upstream_resp.headers.get("content-type")
            )
        except httpx.RequestError as e:
            logger.error(f"连接上游服务器进行非流式传输失败: {e}", exc_info=True)
            return JSONResponse(status_code=502, content={"error": "错误的网关"})


if __name__ == "__main__":
    import uvicorn

    print("正在启动优化后的 Gemini API Python 代理服务器...")
    # 禁用 uvicorn 的默认访问日志，因为我们的中间件提供了更丰富的日志。
    uvicorn.run("main:app", host="0.0.0.0", port=20000, reload=True, access_log=False)
