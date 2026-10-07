FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/home/user/.cache/huggingface

# Hugging Face Spaces требует, чтобы контейнер работал от непривилегированного пользователя
RUN useradd -m -u 1000 user

WORKDIR /app

# torch ставим из CPU-индекса: иначе pip тянет CUDA-сборку на несколько ГБ
COPY requirements.txt .
RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

# Каталоги под БД и кэш модели (в compose на них монтируются тома)
RUN mkdir -p /app/data "$HF_HOME" && chown -R user:user /app/data /home/user/.cache

# Копируем весь код проекта
COPY --chown=user:user . .

# Переключаемся на непривилегированного пользователя
USER user

# Hugging Face Spaces по умолчанию слушает порт 7860
EXPOSE 7860

# Запускаем бота
CMD ["python", "-m", "app.main"]
