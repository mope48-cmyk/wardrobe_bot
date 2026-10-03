# Wardrobe Bot

Telegram-бот для каталогизации одежды и подбора гардероба по погоде.

## Стек
- Python 3.11+
- aiogram 3
- LangGraph
- SQLite (SQLAlchemy)
- Hugging Face (vision)
- OpenWeatherMap

## Установка
1. Создать venv: `python -m venv venv`
2. Активировать: `venv\Scripts\activate`
3. Установить зависимости: `pip install -r requirements.txt`
4. Скопировать `.env.example` в `.env` и заполнить токены
5. Запустить: `python -m app.main`

## Структура
- `app/bot` — Telegram-хендлеры
- `app/graph` — граф LangGraph
- `app/services` — внешние API
- `app/db` — модели и репозитории