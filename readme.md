# Gemini API Python 代理

这是一个优化版的 Gemini API Python 代理服务器，使用 FastAPI 和 HTTPX 构建。它旨在为调用 Google Gemini API 提供一个健壮、可靠且易于观察的中间层。

## ✨ 功能特性

*   **🚀 高性能异步处理**: 基于 FastAPI 和 HTTPX，完全异步，能够处理高并发请求。
*   **🔁 强大的流式重试**: 能够自动处理流式响应中的中断（如 DROP、FINISH\_REASON 异常），并从中断处无缝续传，确保数据流的完整性。
*   **⚙️ Pydantic 配置管理**: 使用 Pydantic 进行配置管理，支持通过 `.env` 文件进行配置，类型安全且易于扩展。
*   **📝 标准化错误响应**: 将上游 API 的错误标准化为统一的 JSON 格式，便于客户端统一处理。
*   **🆔 请求 ID 追踪**: 通过中间件为每个请求注入唯一的 ID，并贯穿整个日志系统，极大地方便了分布式追踪和调试。
*   **📦 Docker 支持**: 提供了 `Dockerfile` 和 `docker-compose.yml`，方便快速部署和容器化。
*   **🔧 灵活的代理**: 可以代理所有到 Gemini API 的请求，包括流式和非流式。

## 🛠️ 技术栈

*   **[FastAPI](https://fastapi.tiangolo.com/)**: 高性能的现代 Web 框架。
*   **[HTTPX](https://www.python-httpx.org/)**: 现代的、异步的 HTTP 客户端。
*   **[Pydantic](https://docs.pydantic.dev/)**: 用于数据验证和设置管理。
*   **[Uvicorn](https://www.uvicorn.org/)**: ASGI 服务器。
*   **[Docker](https://www.docker.com/)**: 容器化平台。

## 🚀 快速开始

### 1. 环境准备

*   Python 3.8+
*   Docker (可选)

### 2. 安装依赖

克隆本项目，然后安装所需的 Python 库：

```bash
git clone https://your-repository-url.com/your-project.git
cd your-project
pip install -r requirements.txt
```

### 3. 配置

在项目根目录创建一个 `.env` 文件。这是存放敏感信息和环境特定配置的地方。

```env
# .env

# 上游 Gemini API 的基础 URL
UPSTREAM_URL_BASE="https://generativelanguage.googleapis.com"

# 最大连续重试次数
MAX_CONSECUTIVE_RETRIES=5

# 是否开启调试模式 (True/False)
DEBUG_MODE=True

# 重试延迟（毫秒）
RETRY_DELAY_MS=1000

# 在重试后是否吞掉 "thought" 块
SWALLOW_THOUGHTS_AFTER_RETRY=True

# 请求超时时间（秒）
REQUEST_TIMEOUT_SECONDS=300
```

### 4. 运行服务

#### 使用 Uvicorn (开发环境)

```bash
uvicorn main:app --host 0.0.0.0 --port 20000 --reload
```

服务器将在 `http://0.0.0.0:20000` 上运行。

#### 使用 Docker (生产环境)

```bash
docker-compose up --build
```

## 📝 API 用法

启动代理服务器后，你可以将原本直接发送到 `https://generativelanguage.googleapis.com` 的请求，改为发送到 `http://localhost:20000`。

**重要**: 请确保在请求头中包含了正确的认证信息，例如 `x-goog-api-key`。

### 示例：调用流式生成模型

```bash
curl -X POST http://localhost:20000/v1beta/models/gemini-pro:streamGenerateContent \
-H "Content-Type: application/json" \
-H "x-goog-api-key: YOUR_API_KEY" \
-d '{
  "contents": [
    {
      "role": "user",
      "parts": [
        {
          "text": "你好，请介绍一下你自己"
        }
      ]
    }
  ]
}'
```

代理服务器会将请求转发到上游，并以流式响应返回结果。如果中途发生可重试的错误，代理会自动处理，客户端对此无感。

## ⚙️ 配置项

所有配置项均可通过 `.env` 文件或环境变量进行设置。

| 变量名                        | 类型    | 默认值                                     | 描述                                           |
| ----------------------------- | ------- | ------------------------------------------ | ---------------------------------------------- |
| `UPSTREAM_URL_BASE`           | `str`   | `https://generativelanguage.googleapis.com` | 上游 Gemini API 的基础 URL。                   |
| `MAX_CONSECUTIVE_RETRIES`     | `int`   | `5`                                        | 流式请求中断后的最大连续重试次数。             |
| `DEBUG_MODE`                  | `bool`  | `True`                                     | 是否开启调试模式，会输出更详细的日志。         |
| `RETRY_DELAY_MS`              | `int`   | `1000`                                     | 每次重试之间的延迟时间（毫秒）。               |
| `SWALLOW_THOUGHTS_AFTER_RETRY`| `bool`  | `True`                                     | 重试后是否忽略模型内部思考过程的 "thought" 块。|
| `REQUEST_TIMEOUT_SECONDS`     | `int`   | `300`                                      | 对上游 API 的请求超时时间（秒）。              |

## 🤝 贡献

欢迎提交 issue 和 pull request！

## 📄 许可证

本项目采用 [MIT](LICENSE) 许可证。