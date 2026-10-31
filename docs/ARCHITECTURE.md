# Архитектура и потоки

## Общая схема слоёв

```
┌──────────────────────────────────────────────────────────────────┐
│                        Telegram (клиент)                        │
└───────────────────────────────┬──────────────────────────────────┘
                                │ Bot API (long polling)
┌───────────────────────────────▼──────────────────────────────────┐
│                      app/bot (aiogram 3)                         │
│  handlers.py  ·  middlewares.py (AlbumMiddleware)                │
│  сборка input_state → вызов графа → _send_ai_message             │
└───────────────────────────────┬──────────────────────────────────┘
                                │ input_state (dict)
┌───────────────────────────────▼──────────────────────────────────┐
│                    app/graph (LangGraph)                         │
│  builder.py: route_condition + узлы из nodes.py                  │
│  state.py: BotState                                             │
└───────────────┬──────────────────────────────┬───────────────────┘
                │                              │
    ┌───────────▼───────────┐      ┌───────────▼───────────┐
    │  app/services          │      │  app/db              │
    │  outfit / llm / vision │      │  models / repository  │
    │  weather / collage /   │      │  SQLite (aiosqlite)  │
    │  gamification / ...    │      └───────────────────────┘
    └───────────┬───────────┘
                │ внешние вызовы
    ┌───────────▼───────────────────────────────────────────────┐
    │ GigaChat · OpenRouter · Open-Meteo · Telegram files API    │
    └────────────────────────────────────────────────────────────┘
```

## Ключевые потоки

### 1. Обычное сообщение / команда
1. `handle_message` (handlers.py) получает сообщение, кладёт в `user_locks` (защита от гонок).
2. Собирает `input_state`: текстовый ввод или фото, `user_id`, `chat_id`, шаг из прошлого состояния.
3. `graph.ainvoke(input_state)` — LangGraph запускает граф; `route_condition` выбирает узел.
4. Узел возвращает `AIMessage(…, additional_kwargs=…)`; `_send_ai_message` отправляет пользователю.

### 2. Фото (новая вещь)
1. Фото попадает в обработчик → `add_photo_node` (или прямой GigaChat-анализ).
2. При успехе распознавания формируется черновик `draft_item`, показывается подтверждение.
3. `add:ai:save` → `_persist_item` сохраняет вещь в БД и вызывает событие геймификации `item_added`.
4. Фоновая генерация товарного фото (`product_photo_bg`) при необходимости.

### 3. Подбор образа
1. `/outfit` → `outfit_start_node` → запрос геолокации (`outfit_location_node`) → `outfit_occasion_node`.
2. `weather.get_weather()` (Open-Meteo) + `outfit.select_outfit()` → варианты.
3. Показ образа: текст + Vogue-коллаж (`vogue_cover`) с кнопками «легче/теплее», «поделиться».

### 4. Утренняя сводка
1. `APScheduler` каждую минуту вызывает `check_morning_pushes` (МСК).
2. Для подходящих пользователей `send_morning_push`: погода → образ → комментарий → коллаж → отправка.
3. Сохраняется `last_sent_date` (защита от дублей).

### 5. Фоновая генерация фото каталога
1. `list_make_product_photo` / `list_toggle_overlay` помечают предмет (`mark_generating`) и добавляют маркер `start_product_photo`.
2. `handlers` создаёт фоновую задачу `generate_product_photo_bg` (в отдельном потоке через `asyncio.to_thread`).
3. По готовности обновляется карточка и `generated_photo_file_id`.

## Планировщик (APScheduler)

| Задача | Расписание (МСК) | Что делает |
| --- | --- | --- |
| `check_morning_pushes` | каждую минуту | ищет пользователей, которым пора отправить сводку |
| `scheduled_broadcast` | 10:00 ежедневно | рассылка `release_notes.txt` |
| `check_old_laundry` | 16:00 ежедневно | напоминание о вещах в стирке >7 дней |

## Клиент LLM с fallback

```
GigaChat (основной)
   │ недоступен / ошибка / пустой ответ
   ▼
OpenRouter (бесплатная модель dots-studio/dots-3-note-preview:free)
   │ недоступен / нет ключа
   ▼
шаблонный текст (генераторы возвращают None/дефолт)
```

Используется в `llm_stylist.py` для комментариев Васильева, свободных ответов и ДНК-профиля.
