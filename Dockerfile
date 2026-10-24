# syntax=docker/dockerfile:1.4

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/home/user/.cache/huggingface

RUN useradd -m -u 1000 user

RUN apt-get update && apt-get install -y --no-install-recommends \
    fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

# 1. torch + torchvision из CPU-индекса PyTorch (без CUDA)
# Кэш pip сохраняется между сборками через cache mount — упавшие
# скачивания не теряются при повторных попытках.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install \
    --index-url https://download.pytorch.org/whl/cpu \
    torch torchvision

# 2. Остальные зависимости — из Aliyun, тоже с cache mount
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install \
    --index-url https://mirrors.aliyun.com/pypi/simple/ \
    --trusted-host mirrors.aliyun.com \
    --timeout 900 \
    --retries 50 \
    -r requirements.txt

RUN mkdir -p /app/data "$HF_HOME" && chown -R user:user /app/data /home/user/.cache

COPY --chown=user:user . .

USER user

EXPOSE 7860

CMD ["python", "-m", "app.main"]