# 使用官方 Python 运行时作为父镜像
FROM python:3.10-slim-buster

# 设置工作目录
WORKDIR /app

# 将依赖文件复制到工作目录
COPY requirements.txt .

# 安装所需的包
RUN pip install --no-cache-dir -r requirements.txt

# 将当前目录内容复制到容器的 /app 目录
COPY . .

# 暴露端口，让容器外可以访问
EXPOSE 20000

# 运行 app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "20000"]