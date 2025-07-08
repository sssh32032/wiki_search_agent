# ---- 基礎映像 ----
FROM python:3.12-slim

# ---- 系統依賴 ----
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# ---- Poetry 安裝 ----
ENV POETRY_VERSION=1.8.2
RUN pip install "poetry==$POETRY_VERSION"

# ---- 專案檔案 ----
WORKDIR /app
COPY . /app

# ---- 安裝依賴 ----
RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi --only main

# ---- 複製 .env ----
COPY env.example .env

# ---- 開放 API 端口 ----
EXPOSE 8000

# ---- 啟動 API ----
CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000"] 