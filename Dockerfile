FROM python:3.11-slim

# Устанавливаем системные зависимости
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Hugging Face Spaces требует, чтобы контейнер работал от непривилегированного пользователя
RUN useradd -m -u 1000 user

WORKDIR /app

# Копируем и устанавливаем Python-зависимости
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем весь код проекта
COPY --chown=user:user . .

# Переключаемся на непривилегированного пользователя
USER user

# Hugging Face Spaces по умолчанию слушает порт 7860
EXPOSE 7860

# Запускаем бота
CMD ["python", "-m", "app.main"]