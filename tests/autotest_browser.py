# autotest_browser.py
# Автотест @MyWordroobBot через Telegram Web (web.telegram.org/a/).
#
# Сценарий для каждого фото:
#   b. отправить одно фото из папки (оригинальный файл с расширением)
#   c. дождаться ответа (он обязательно будет)
#   d. зафиксировать ответ + время в CSV
#   e. отправить /cancel
#   → к следующему фото
#
# Требует: pip install playwright && python -m playwright install chromium
#
# Запуск:
#   python autotest_browser.py            # обычный прогон
#   python autotest_browser.py --debug    # дамп кнопок перед стартом
#   python autotest_browser.py --fresh    # игнорировать сохранённую сессию

import asyncio
import csv
import sys
import time
from datetime import datetime
from pathlib import Path
from playwright.async_api import async_playwright, Page

# === Конфигурация ===
IMAGES_DIR    = Path(r"C:/Users/User/Desktop/Гардероб/clothes_images")
CSV_PATH      = Path(r"C:/Projects/wardrobe_bot/tests/bot_test_results_with_urls.csv")
MAX_REPLY_SEC = 60
PAUSE_SEC     = 2
STATE_FILE    = Path("tg_state.json")

# Команда, которую нужно отправить ПЕРЕД каждым фото.
# Если бот требует /add, чтобы войти в режим добавления — впиши "/add".
# Если не нужно — оставь пустую строку.
PRE_PHOTO_COMMAND = ""     # например: "/add"

DEBUG = "--debug" in sys.argv
FRESH = "--fresh" in sys.argv

# === Селекторы Telegram Web A ===
SEL_MESSAGE_INPUT = "#editable-message-text, .input-message-input[contenteditable='true']"
SEL_MESSAGES      = ".Message, .message"
SEL_MODAL_CLOSE   = "button.close-button"

ATTACH_BUTTON_CANDIDATES = [
    "button[aria-label='Add an attachment']",
    "button[aria-label='Прикрепить файл']",
    ".AttachMenu--button",
    "button[title='Add an attachment']",
]

ATTACH_MENU_LABELS = [
    "Photo or Video",
    "Фото или видео",
    "Photo",
    "Фото",
]

SEND_BUTTON_CANDIDATES = [
    ".SendFileModal .btn-send",
    ".modal .btn-send",
    "[class*='SendFile'] .btn-send",
    ".btn-send",
    "button.btn-send",
    "button[aria-label='Send']",
    "button[aria-label='Отправить']",
]

CSV_FIELDS = [
    "num", "file", "source_url", "category", "type", "color",
    "material", "check", "sent_at", "response", "response_type",
    "response_time_sec",
]


# ---------- CSV / метаданные ----------

def load_existing_meta():
    meta = {}
    if not CSV_PATH.exists():
        return meta
    try:
        with open(CSV_PATH, encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                if row.get("file"):
                    meta[row["file"]] = row
    except Exception as e:
        print(f">>> ⚠ Не удалось прочитать существующий CSV: {e}")
    return meta


def parse_filename(filename):
    stem = Path(filename).stem
    parts = stem.split("_")
    if len(parts) < 5:
        return {}
    return {
        "num": parts[0],
        "category": parts[1],
        "type": parts[2],
        "color": parts[3],
        "material": parts[4],
        "check": "_".join(parts[5:]) if len(parts) > 5 else "",
    }


def save_csv(rows):
    try:
        with open(CSV_PATH, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=CSV_FIELDS, quoting=csv.QUOTE_ALL
            )
            writer.writeheader()
            for row in rows:
                writer.writerow({k: row.get(k, "") for k in CSV_FIELDS})
        print(f">>> CSV сохранён: {CSV_PATH}")
    except Exception as e:
        print(f">>> ⚠ Ошибка сохранения CSV: {e}")


# ---------- основной сценарий ----------

async def main():
    t_start = time.time()
    existing_meta = load_existing_meta()
    print(f">>> Загружено строк из старого CSV: {len(existing_meta)}")

    if not IMAGES_DIR.exists():
        print(f">>> ❌ Папка не найдена: {IMAGES_DIR}")
        return

    images = sorted(
        p for p in IMAGES_DIR.rglob("*")
        if p.is_file() and p.suffix.lower() in (".png", ".jpg", ".jpeg")
    )
    if not images:
        print(f">>> ❌ В папке нет изображений: {IMAGES_DIR}")
        return
    print(f">>> Найдено изображений: {len(images)}")

    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=False,
            args=["--start-maximized"],
        )
        storage = None
        if not FRESH and STATE_FILE.exists():
            storage = str(STATE_FILE)
            print(">>> Использую сохранённую сессию")
        context = await browser.new_context(
            no_viewport=True,
            storage_state=storage,
        )
        page = await context.new_page()

        print(">>> Открываю https://web.telegram.org/a/")
        if not await open_telegram(page):
            await browser.close()
            return

        if not await wait_for_start_command():
            print(">>> Отменено пользователем.")
            await context.storage_state(path=str(STATE_FILE))
            await browser.close()
            return

        try:
            await page.wait_for_selector(SEL_MESSAGE_INPUT, timeout=5000)
        except Exception:
            print(">>> ⚠ Не вижу поле ввода сообщения — продолжаю через 3 сек")
            await asyncio.sleep(3)

        await context.storage_state(path=str(STATE_FILE))
        await close_modals(page)

        if DEBUG:
            await debug_dump(page, "BEFORE START")

        results_rows = []

        for idx, img_path in enumerate(images, 1):
            filename = img_path.name
            print(f"[{idx}/{len(images)}] {filename}")
            try:
                row = await process_one_photo(
                    page, img_path, existing_meta
                )
            except Exception as e:
                print(f"  ❌ Ошибка: {e}")
                row = error_row(filename, existing_meta, f"[ERROR] {e}")

            results_rows.append(row)

            if row["response"] and not row["response"].startswith("[ERROR]"):
                print(f"  ✅ Ответ за {row['response_time_sec']} сек")
            elif row["response"]:
                print(f"  ❌ {row['response'][:80]}")
            else:
                print("  ⏱ Таймаут ответа")

            # e. /cancel после каждого фото
            await send_cancel(page)
            await asyncio.sleep(PAUSE_SEC)

        save_csv(results_rows)

        total = len(results_rows)
        ok = sum(
            1 for r in results_rows
            if r["response"] and not r["response"].startswith("[ERROR]")
        )
        times = [
            float(r["response_time_sec"])
            for r in results_rows if r["response_time_sec"]
        ]
        avg = round(sum(times) / len(times), 2) if times else 0
        print("\n=== ИТОГО ===")
        print(f"Всего фото:          {total}")
        print(f"Успешных ответов:    {ok}")
        print(f"Средняя задержка:    {avg} сек")
        print(f"Общее время прогона: {time.time() - t_start:.0f} сек")
        print(f"CSV: {CSV_PATH}")

        await context.storage_state(path=str(STATE_FILE))
        await browser.close()


async def process_one_photo(page: Page, img_path: Path,
                            existing_meta: dict) -> dict:
    """b→c→d для одного фото."""
    filename = img_path.name
    meta = existing_meta.get(filename) or parse_filename(filename)

    await close_modals(page)

    # Опциональная команда перед фото (например, /add)
    if PRE_PHOTO_COMMAND:
        try:
            await page.click(SEL_MESSAGE_INPUT, timeout=2000)
            await page.keyboard.type(PRE_PHOTO_COMMAND)
            await page.keyboard.press("Enter")
            await asyncio.sleep(1.5)
        except Exception as e:
            print(f"  ⚠ PRE_PHOTO_COMMAND не отправлен: {e}")

    # Снимок счётчика ДО отправки
    prev_count = await safe_count_messages(page)

    sent_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    t0 = time.time()

    # b. Отправляем оригинальный файл (с расширением!)
    if not await send_photo(page, img_path):
        return error_row(filename, existing_meta,
                         "[ERROR] upload failed", sent_at=sent_at)

    # c. Ждём ответа
    response_text, elapsed = await wait_for_reply(
        page, prev_count, MAX_REPLY_SEC
    )

    # d. Строка CSV
    return {
        "num": meta.get("num", ""),
        "file": filename,
        "source_url": meta.get("source_url", ""),
        "category": meta.get("category", ""),
        "type": meta.get("type", ""),
        "color": meta.get("color", ""),
        "material": meta.get("material", ""),
        "check": meta.get("check", ""),
        "sent_at": sent_at,
        "response": response_text or "",
        "response_type": "text" if response_text else "timeout",
        "response_time_sec": round(elapsed, 2) if elapsed else "",
    }


def error_row(filename: str, existing_meta: dict, msg: str,
              sent_at: str | None = None) -> dict:
    meta = existing_meta.get(filename) or parse_filename(filename)
    return {
        "num": meta.get("num", ""),
        "file": filename,
        "source_url": meta.get("source_url", ""),
        "category": meta.get("category", ""),
        "type": meta.get("type", ""),
        "color": meta.get("color", ""),
        "material": meta.get("material", ""),
        "check": meta.get("check", ""),
        "sent_at": sent_at or datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "response": msg,
        "response_type": "error",
        "response_time_sec": "",
    }


# ---------- работа с браузером ----------

async def open_telegram(page: Page) -> bool:
    for attempt in range(5):
        try:
            await page.goto(
                "https://web.telegram.org/a/",
                wait_until="domcontentloaded",
                timeout=60000,
            )
            return True
        except Exception as e:
            print(f">>> goto попытка {attempt + 1}: {e}")
            await asyncio.sleep(3)
    return False


async def wait_for_start_command() -> bool:
    print()
    print("=" * 60)
    print("  РУЧНАЯ АВТОРИЗАЦИЯ И ВЫБОР ЧАТА С БОТОМ")
    print("=" * 60)
    print("  1. Войди в Telegram Web в открытом окне.")
    print("  2. Открой чат с @MyWordroobBot.")
    print("  3. Убедись, что видно поле ввода сообщения.")
    print("  4. Напиши 'start' здесь и нажми Enter.")
    print("  ('q' — выход)")
    print("=" * 60)
    loop = asyncio.get_event_loop()
    while True:
        cmd = (await loop.run_in_executor(None, input, ">>> ")).strip().lower()
        if cmd in ("start", "s", "го"):
            return True
        if cmd in ("q", "quit", "exit", "выход"):
            return False
        print("Напиши 'start' или 'q'.")


async def close_modals(page: Page):
    for _ in range(3):
        try:
            btn = await page.query_selector(SEL_MODAL_CLOSE)
            if btn and await btn.is_visible():
                try:
                    await page.click(SEL_MODAL_CLOSE, timeout=1500)
                except Exception:
                    await page.keyboard.press("Escape")
                await page.wait_for_timeout(300)
            else:
                return
        except Exception:
            return


async def safe_count_messages(page: Page) -> int:
    try:
        return await page.eval_on_selector_all(
            SEL_MESSAGES, "els => els.length"
        )
    except Exception:
        return 0


async def send_photo(page: Page, path: Path) -> bool:
    """
    Отправляет оригинальный файл из папки.
    Файл сохраняет расширение (.jpg/.png), поэтому Telegram шлёт его как фото,
    а не как документ.
    """
    await close_modals(page)

    if not path.exists():
        print(f"  ⚠ файл не найден: {path}")
        return False

    # 1. Скрепка
    opened = False
    for sel in ATTACH_BUTTON_CANDIDATES:
        try:
            await page.click(sel, timeout=2000)
            opened = True
            print(f"  · открыл меню вложений: {sel}")
            break
        except Exception:
            continue
    if not opened:
        print("  ⚠ скрепка не найдена")
        return False

    await page.wait_for_timeout(500)

    # 2. «Photo or Video» → file chooser
    try:
        async with page.expect_file_chooser(timeout=8000) as fc_info:
            clicked = False
            for label in ATTACH_MENU_LABELS:
                try:
                    await page.click(f"text={label}", timeout=1500)
                    clicked = True
                    print(f"  · выбрал пункт: {label}")
                    break
                except Exception:
                    continue
            if not clicked:
                print("  ⚠ пункт 'Photo or Video' не найден")
                await page.keyboard.press("Escape")
                return False
        chooser = await fc_info.value
        await chooser.set_files(str(path))
    except Exception as e:
        print(f"  ⚠ file chooser: {e}")
        return False

    # 3. Ждём превью
    await page.wait_for_timeout(2500)

    # 4. Отправка: сначала пробуем кнопку, потом Enter
    for sel in SEND_BUTTON_CANDIDATES:
        try:
            btn = await page.query_selector(sel)
            if btn and await btn.is_visible():
                await btn.click()
                print(f"  · нажал кнопку отправки: {sel}")
                return True
        except Exception:
            continue

    try:
        await page.keyboard.press("Enter")
        print("  · отправил через Enter")
        return True
    except Exception as e:
        print(f"  ⚠ не смог отправить: {e}")
        return False


async def wait_for_reply(page: Page, prev_count: int, timeout: float):
    """
    Ждёт, пока в DOM появятся минимум 2 новых сообщения:
    наше фото + ответ бота. Возвращает (текст последнего, время).
    """
    start = time.time()
    while time.time() - start < timeout:
        try:
            cur = await page.eval_on_selector_all(
                SEL_MESSAGES, "els => els.length"
            )
        except Exception:
            await asyncio.sleep(0.5)
            continue

        if cur >= prev_count + 2:
            await asyncio.sleep(0.6)  # дать боту дорисовать
            text = await get_last_message_text(page)
            return text, time.time() - start

        await asyncio.sleep(0.4)

    return None, None


async def get_last_message_text(page: Page) -> str:
    try:
        return await page.evaluate(
            """() => {
                const msgs = document.querySelectorAll('.Message, .message');
                if (msgs.length === 0) return '';
                const last = msgs[msgs.length - 1];
                const textEl = last.querySelector(
                    '.text-content, .message-text, .text'
                );
                const target = textEl || last;
                return (target.innerText || '').trim();
            }"""
        )
    except Exception as e:
        print(f"  ⚠ get_last_message_text: {e}")
        return ""


async def send_cancel(page: Page):
    try:
        await close_modals(page)
        await page.click(SEL_MESSAGE_INPUT, timeout=2000)
        await page.keyboard.type("/cancel")
        await page.keyboard.press("Enter")
    except Exception as e:
        print(f"  ⚠ /cancel: {e}")


async def debug_dump(page: Page, title: str = ""):
    print(f"\n=== DEBUG: {title} ===")
    buttons = await page.eval_on_selector_all(
        "button",
        """els => els.map(e => ({
            title: e.title || '',
            aria: e.getAttribute('aria-label') || '',
            cls: (e.className||'').toString().slice(0,80),
            text: (e.innerText||'').slice(0,30)
        }))""",
    )
    for b in buttons:
        print(b)
    print(f"=== /DEBUG ===\n")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n>>> Прервано пользователем.")