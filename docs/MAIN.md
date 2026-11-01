# Wardrobe Bot — Техническая документация

Telegram-бот «Гардероб» с ИИ-стилистом (`@MyWordroobBot`).
Стек: Python 3.12+, aiogram 3.x, LangGraph, SQLAlchemy (async) + SQLite,
APScheduler, Pillow, HuggingFace-модели (SegFormer/SigLIP), rembg, GigaChat,
OpenRouter (fallback), Open-Meteo.

---

## Оглавление

- [Архитектура и потоки](ARCHITECTURE.md)
- [База данных](DATABASE.md)
- [Расположение и описание всех файлов](#полный-список-файлов)
- [Диаграммы](diagrams/):
  - [Обзор архитектуры](diagrams/overview.html)
  - [Граф состояний LangGraph](diagrams/graph_flow.html)
  - [Схема базы данных](diagrams/database.html)
  - [Деплой (Docker)](diagrams/deployment.html)
  - [Алгоритм подбора одежды](outfit-algorithm.html)

---

## Полный список файлов

### Корень проекта

| Файл | Назначение |
| --- | --- |
| `app/main.py` | **Точка входа.** Инициализация БД, создание `Bot`/`Dispatcher`, подключение роутеров и middleware. Запуск `APScheduler`: проверка утренних сводок каждую минуту (МСК), ежедневная рассылка `release_notes.txt` в 10:00, напоминание о «старой стирке» в 16:00. Прогрев моделей (`warmup`). Старт long-polling. |
| `app/config.py` | Конфигурация: dataclass `Settings`, читается из `.env` (через `python-dotenv`). Ключи: `BOT_TOKEN`, `DATABASE_URL`, `GIGACHAT_CREDENTIALS`, `OPENROUTER_API_KEY`, `OWM_API_KEY` (оставлен для совместимости), и др. |
| `requirements.txt` | Боевые зависимости (aiogram, langgraph, langchain-gigachat, SQLAlchemy, rembg, torch, transformers, Pillow, аpscheduler …). |
| `requirements-dev.txt` | Лёгкие зависимости для CI/тестов (pytest, ruff, без тяжёлых ML-модулей). |
| `pyproject.toml` | Конфиг `ruff` (линтер) с правилами и осознанными игнорами (русский текст и т.п.). |
| `pytest.ini` | Настройки pytest (`asyncio_mode = auto`, `testpaths = tests`). |
| `Dockerfile` | Сборка образа: Python 3.12, torch из зеркала SJTU, установка зависимостей, копирование кода, запуск `python -m app.main`. |
| `docker-compose.yml` | Сервис `bot` с томом для `data/` (SQLite), переменными окружения из `.env`. |
| `.env.example` | Образец секретов (без значений). |
| `README.md` | Пользовательское описание бота. |
| `release_notes.txt` | Тексты «релизных новостей» для рассылки (HTML). |

### `app/db/` — слой данных

| Файл | Назначение |
| --- | --- |
| `database.py` | Async-движок SQLAlchemy (`sqlite+aiosqlite`), фабрика сессий `SessionLocal`, `init_db()` (создаёт таблицы через `Base.metadata.create_all`). |
| `models.py` | Модели: `WardrobeItem` (вещь), `UserMorningSettings` (утренняя сводка), `UserPreferences` (дизлайки), `UserGamification` (XP, серия, достижения, вызов недели). |
| `repository.py` | Функции доступа к данным: CRUD вещей (`add_item`, `get_items`, `update_item_field` с белым списком полей, `delete_item`), настройки утренней сводки, дизлайки, префиксы геймификации. |
| `migrations.py` | Скрипты миграций SQLite (добавление колонок), совместимые со старой БД. |

### `app/bot/` — интерфейс Telegram

| Файл | Назначение |
| --- | --- |
| `handlers.py` | Роутеры aiogram: `@router.message()` (текст/фото/альбомы) и `@router.callback_query()`. Собирают `input_state` и вызывают граф LangGraph. `_send_ai_message` отвечает за отправку: фото, медиагруппы, редактирование caption, inline-клавиатуры, HTML-санитайзинг. Запускает фоновую генерацию фото каталога (`start_product_photo`). |
| `middlewares.py` | `AlbumMiddleware` — собирает медиагруппы (несколько фото) в один список за 0.5 с. |

### `app/graph/` — логика диалога (LangGraph)

| Файл | Назначение |
| --- | --- |
| `state.py` | `BotState` — `TypedDict`: сообщения, user_id, шаг (`step`), интент, черновик вещи, параметры листа/образа/капсулы и т.д. |
| `builder.py` | Сборка `StateGraph`: регистрация узлов, `route_condition` (маршрутизация по callback/тексту), `_match_free_text_intent` (распознавание намерений свободного текста), список терминальных узлов. |
| `nodes.py` | Все узлы-обработчики: онбординг (`start_node`), добавление вещи (`add_*`), просмотр/редактирование гардероба (`list_*`), подбор образа (`outfit_*`), приговор (`verdict_*`), ДНК (`dna_*`), утренняя сводка (`morning_*`), капсула (`capsule_*`), стирка (`laundry_*`), геймификация (`progress_node`). Плюс внутренние хелперы. |

### `app/services/` — бизнес-логика

| Файл | Назначение |
| --- | --- |
| `outfit.py` | Математический алгоритм подбора комплекта: правила погоды/сезона/слоёв, скор вещей (толщина, капюшон, вода), подбор аксессуаров, fallback-алгоритм. |
| `llm_stylist.py` | Генерация комментариев «Васильева» (для образа, свободных ответов, ДНК). Провайдеры с fallback: **GigaChat → OpenRouter → шаблон**. |
| `vision_gigachat.py` | Мультимодальный анализ фото вещи через GigaChat-2-Pro (категория, цвет, материал, тепло, слой …). |
| `vision.py` | Локальное распознавание одежды: SigLIP (классификация) + HSV-анализ цветов. |
| `segmentation.py` | Сегментация одежды SegFormer (маска вещи на фото). |
| `product_photo.py` | Товарное фото для каталога: rembg + SegFormer, композит на белом фоне, «белый слой» по выбору пользователя. |
| `product_photo_bg.py` | Фоновая генерация фото каталога (в отдельном потоке), обновление карточки. |
| `weather.py` | Погода и геокодинг через **Open-Meteo** (без ключа): `geocode_city`, `get_weather`. |
| `morning_push.py` | Сборка утренней сводки: погода → подбор образа → комментарий → Vogue-коллаж → отправка; учитывает серию дней. |
| `vogue_cover.py` | Генерация Vogue-коллажа (Pillow): сетка фото, подписи, шапка с погодой/датой. |
| `dna_card.py` | Отрисовка «ДНК-карточки» пользователя (стиль, цвета, тепло, пробелы). |
| `style_dna.py` | Детерминированный анализ гардероба (без LLM): доминирующий стиль, палитра, нейтральность. |
| `verdict.py` | «Приговор» по фото образа (мультимодальная GigaChat). |
| `capsule.py` | Алгоритм капсульного гардероба: прогон по сценариям погоды, частотность, баланс категорий. |
| `gamification.py` | Геймификация: XP/уровни (`100*level^1.5`), серия дней, 14 достижений, недельный детерминированный вызов. |
| `html_sanitizer.py` | Санитайзинг HTML-ответов LLM перед отправкой (разрешённые теги, экранирование). |
| `telegram_files.py` | Скачивание фото из Telegram Bot API по `file_id` (httpx). |
| `broadcast.py` | Рассылка сообщений всем пользователям (например, релизных новостей). |

### `tests/` — тесты

| Файл | Назначение |
| --- | --- |
| `conftest.py` | Стубы тяжёлых ML-модулей (torch/transformers/rembg) + временная SQLite-БД. |
| `test_outfit.py` | Математический алгоритм подбора (погода, слои, зонт, толщина, капюшон). |
| `test_capsule.py` | Капсула: фильтры, лимиты категорий, сортировка. |
| `test_builder_routing.py` | Интенты свободного текста и маршрутизация callback. |
| `test_onboarding.py` | Онбординг: прогресс-бар, подсказки, «праздник» на 5 вещей. |
| `test_repository.py` | CRUD и слои БД на временной базе. |
| `test_gamification.py` | XP, уровни, серия, достижения, недельный вызов. |
| `test_llm_stylist.py` | Fallback-цепочка LLM: GigaChat → OpenRouter → шаблон. |
| `test_html_sanitizer.py` | Чистка HTML. |
| `test_weather.py` | Маппинг погоды Open-Meteo. |
