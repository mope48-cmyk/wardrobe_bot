# База данных

Движок: **SQLite** (async, `aiosqlite`). Файл: `data/wardrobe.db` (на HF Spaces — `/data/wardrobe.db`).
Таблицы создаются автоматически при старте (`init_db`); для существующей БД предусмотрены ручные миграции (`app/db/migrations.py`).

## Таблицы

### `wardrobe_items` — вещи пользователя

| Поле | Тип | Назначение |
| --- | --- | --- |
| `id` | INTEGER PK | id вещи |
| `user_id` | INTEGER idx | владелец (Telegram id) |
| `photo_file_id` | STR | file_id оригинального фото |
| `generated_photo_file_id` | STR NULL | file_id товарного фото (каталог) |
| `category` | STR | верх / низ / обувь / аксессуар |
| `type` | STR | тип (футболка, джинсы…) |
| `colors` | TEXT (JSON) | 1–3 цвета |
| `material` | STR | материал |
| `warmth_level` | INT | тепло: 1–5 |
| `waterproof` | BOOL | водонепроницаемость |
| `thickness` | INT | толщина: 1–5 |
| `has_hood` | BOOL | капюшон |
| `use_white_overlay` | BOOL | принудительный белый фон |
| `formal_level` | STR | casual / business / sport |
| `season` | STR | лето / демисезон / зима / универсальная |
| `layer` | STR | base / mid / outer / none |
| `accessory_occasions` | TEXT (JSON) | поводы для аксессуара |
| `is_in_laundry` | BOOL | вещь в стирке |
| `laundry_start_date` | DATETIME NULL | когда отправили в стирку |
| `created_at` | DATETIME | дата добавления |

### `user_morning_settings` — настройки утренней сводки

`user_id` PK · `enabled` · `time_str` ("HH:MM") · `city` · `lat` · `lon` · `occasion` · `days` (CSV: 0=Пн…6=Вс) · `last_sent_date` (YYYY-MM-DD).

### `user_preferences` — предпочтения

`user_id` PK · `disliked_items` (TEXT JSON: id исключённых вещей).

### `user_gamification` — геймификация

`user_id` PK · `xp` · `streak` · `last_active_date` · `weekly_challenge` · `challenge_progress` · `challenge_target` · `challenge_week` · `achievements` (TEXT JSON) · счётчики: `items_added`, `outfits_created`, `verdicts_done`, `capsules_built`, `laundry_done`.

## Замечания

- **Белый список полей** в `update_item_field` защищает от случайной перезаписи `id/user_id/created_at`.
- **Миграции**: новые колонки добавляются точечными `ALTER TABLE` (исторически: `use_white_overlay`, `thickness`, `has_hood`, `laundry_start_date`, геймификация — автоматически через `create_all`).
- Дата «сегодня» всегда считается по **Москве** (`Europe/Moscow`).
