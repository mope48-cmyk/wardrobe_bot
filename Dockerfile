FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/home/user/.cache/huggingface

RUN useradd -m -u 1000 user

WORKDIR /app

COPY requirements.txt .

# 1. torch + torchvision — из CPU-индекса PyTorch (без CUDA, ~200 МБ вместо ~1.3 ГБ)
RUN pip install \
    --index-url https://download.pytorch.org/whl/cpu \
    torch torchvision

# 2. Остальные зависимости — из зеркала Aliyun (без CUDA-зависимостей)
RUN pip install \
    --index-url https://mirrors.aliyun.com/pypi/simple/ \
    --trusted-host mirrors.aliyun.com \
    -r requirements.txt

# 3. Каталоги под БД и кэш модели
RUN mkdir -p /app/data "$HF_HOME" && chown -R user:user /app/data /home/user/.cache

COPY --chown=user:user . .

USER user

EXPOSE 7860

CMD ["python", "-m", "app.main"]