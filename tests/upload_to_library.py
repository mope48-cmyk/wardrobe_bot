# upload_to_library.py
# Автоматическая загрузка вещей в библиотеку чат-бота через Telegram Web A.
#
# Требует: pip install playwright && python -m playwright install chromium

import asyncio
import re
import sys
import time
from pathlib import Path
from playwright.async_api import async_playwright, Page

# === Конфигурация ===
IMAGES_DIR    = Path(r"C:/Users/User/Desktop/Гардероб/clothes_images")
MAX_REPLY_SEC = 15       # Время ожидания ответа от бота
PAUSE_SEC     = 2
STATE_FILE    = Path("tg_state.json")

# С какой по счёту картинки начинать
START_FROM    = 16

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

# ---------- Основной сценарий ----------

async def main():
    if not IMAGES_DIR.exists():
        print(f">>> ❌ Папка не найдена: {IMAGES_DIR}")
        return

    images = sorted(
        p for p in IMAGES_DIR.rglob("*")
        if p.is_file() and p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp")
    )
    if not images:
        print(f">>> ❌ В папке нет изображений: {IMAGES_DIR}")
        return

    images_to_process = images[START_FROM - 1:]
    if not images_to_process:
        print(f">>> ❌ В папке меньше {START_FROM} изображений.")
        return

    print(f">>> Всего в папке: {len(images)} шт.")
    print(f">>> Пропускаю первые {START_FROM - 1} фото.")
    print(f">>> Начинаю с {START_FROM}-го: {images_to_process[0].name}")

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

        t_start = time.time()
        success_count = 0
        fail_count = 0

        for idx, img_path in enumerate(images_to_process, START_FROM):
            filename = img_path.name
            print(f"\n[{idx}/{len(images)}] {filename}")
            
            try:
                await process_one_photo(page, img_path)
                success_count += 1
                print(f"  ✅ Успешно добавлено в библиотеку")
            except Exception as e:
                fail_count += 1
                print(f"  ❌ Ошибка: {e}")
                await send_text(page, "/cancel")
                await asyncio.sleep(2)

        print("\n=== ИТОГО ===")
        print(f"Обработано фото:     {success_count + fail_count}")
        print(f"Успешно загружено:   {success_count}")
        print(f"Ошибок:              {fail_count}")
        print(f"Общее время:         {time.time() - t_start:.0f} сек")

        await context.storage_state(path=str(STATE_FILE))
        await browser.close()


async def process_one_photo(page: Page, img_path: Path):
    await close_modals(page)

    # 0. СБРОС
    await send_text(page, "/cancel")
    await asyncio.sleep(2.0)

    # 1. Отправляем фото
    count_before = await get_msg_count(page)
    if not await send_photo(page, img_path):
        raise RuntimeError("Не удалось отправить фото")
    
    # 2. Ждём распознавание
    text = await wait_for_new_text(page, "Я распознал вещь", count_before, MAX_REPLY_SEC)
    if not text:
        raise RuntimeError("Бот не прислал распознавание")
    
    guessed_type = extract_guessed_type(text)
    print(f"  · Бот предположил тип: {guessed_type}")
    
    # 3. Отправляем тип
    count_before = await get_msg_count(page)
    await send_text(page, guessed_type)
    
    # 4. ДИАЛОГОВЫЙ ЦИКЛ (макс 10 шагов)
    for step in range(10):
        text = await wait_for_new_message(page, count_before, MAX_REPLY_SEC)
        if not text:
            raise RuntimeError("Бот не ответил вовремя (таймаут)")

        # Если бот закончил сохранение — выходим из цикла
        if "Готово! Сохранил вещь:" in text:
            print("  · Бот подтвердил сохранение.")
            return

        count_before = await get_msg_count(page)

        # Анализируем вопрос бота и отвечаем
        if "Похоже, вещь цвета" in text or "Всё верно?" in text:
            print("  · Отправляю: 'Подтвердить'")
            await send_text(page, "Подтвердить")
            
        elif "Предположительно:" in text:
            layer = extract_guessed_layer(text)
            print(f"  · Бот предположил слой: {layer}")
            await send_text(page, layer)
            
        elif "материал" in text.lower() or "из чего" in text.lower():
            print("  · Отправляю материал: Хлопок")
            await send_text(page, "Хлопок")
            
        else:
            print(f"  · ⚠ Неизвестный текст от бота: {text[:60]}...")
            # Аварийный ответ, чтобы не зависнуть
            await send_text(page, "Хлопок")

    raise RuntimeError("Слишком много шагов, не дождались 'Готово!'")


# ---------- Взаимодействие с браузером ----------

async def get_msg_count(page: Page) -> int:
    try:
        return await page.eval_on_selector_all(SEL_MESSAGES, "els => els.length")
    except Exception:
        return 0

async def wait_for_new_message(page: Page, min_count: int, timeout: float) -> str:
    """Ждёт появления любого нового сообщения (текста)."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            current_count = await get_msg_count(page)
            if current_count > min_count:
                await asyncio.sleep(0.5) # Ждем отрисовки
                all_texts = await page.locator(SEL_MESSAGES).all_inner_texts()
                new_texts = all_texts[min_count:]
                combined = "\n".join(new_texts)
                if combined.strip():
                    return combined
        except Exception:
            pass
        await asyncio.sleep(0.3)
    return ""

async def wait_for_new_text(page: Page, keyword: str, min_count: int, timeout: float) -> str:
    """Ждёт новое сообщение с конкретным текстом."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            current_count = await get_msg_count(page)
            if current_count > min_count:
                await asyncio.sleep(0.5)
                all_texts = await page.locator(SEL_MESSAGES).all_inner_texts()
                new_texts = all_texts[min_count:]
                combined = "\n".join(new_texts)
                if keyword in combined:
                    return combined
        except Exception:
            pass
        await asyncio.sleep(0.3)
    return ""


async def send_text(page: Page, text: str):
    await close_modals(page)
    await page.click(SEL_MESSAGE_INPUT, timeout=3000)
    await page.keyboard.type(text)
    await page.keyboard.press("Enter")
    await asyncio.sleep(0.5)


async def send_photo(page: Page, path: Path) -> bool:
    if not path.exists():
        print(f"  ⚠ файл не найден: {path}")
        return False

    opened = False
    for sel in ATTACH_BUTTON_CANDIDATES:
        try:
            await page.click(sel, timeout=2000)
            opened = True
            break
        except Exception:
            continue
    if not opened:
        print("  ⚠ скрепка не найдена")
        return False

    await page.wait_for_timeout(500)

    try:
        async with page.expect_file_chooser(timeout=8000) as fc_info:
            clicked = False
            for label in ATTACH_MENU_LABELS:
                try:
                    await page.click(f"text={label}", timeout=1500)
                    clicked = True
                    break
                except Exception:
                    continue
            if not clicked:
                await page.keyboard.press("Escape")
                return False
        chooser = await fc_info.value
        await chooser.set_files(str(path))
    except Exception as e:
        print(f"  ⚠ file chooser: {e}")
        return False

    await page.wait_for_timeout(2500)

    for sel in SEND_BUTTON_CANDIDATES:
        try:
            btn = await page.query_selector(sel)
            if btn and await btn.is_visible():
                await btn.click()
                return True
        except Exception:
            continue

    try:
        await page.keyboard.press("Enter")
        return True
    except Exception as e:
        return False


# ---------- Парсинг ----------

def extract_guessed_type(text: str) -> str:
    match = re.search(r"^1\.\s*(.+?)\s*[—\-]", text, re.MULTILINE)
    if match:
        return match.group(1).strip()
    raise ValueError("Не смог найти тип в ответе бота")


def extract_guessed_layer(text: str) -> str:
    match = re.search(r"Предположительно:\s*([^\.\n]+)", text)
    if match:
        return match.group(1).strip()
    raise ValueError("Не смог найти слой в ответе бота")


# ---------- Утилиты ----------

async def open_telegram(page: Page) -> bool:
    for attempt in range(5):
        try:
            await page.goto("https://web.telegram.org/a/", wait_until="domcontentloaded", timeout=60000)
            return True
        except Exception as e:
            print(f">>> goto попытка {attempt + 1}: {e}")
            await asyncio.sleep(3)
    return False


async def wait_for_start_command() -> bool:
    print("\n" + "=" * 60)
    print("  Войди в Telegram Web и открой чат с @MyWordroobBot.")
    print("  Затем напиши 'start' и нажми Enter.")
    print("=" * 60)
    loop = asyncio.get_event_loop()
    while True:
        cmd = (await loop.run_in_executor(None, input, ">>> ")).strip().lower()
        if cmd in ("start", "s", "го"): return True
        if cmd in ("q", "quit", "выход"): return False
        print("Напиши 'start' или 'q'.")


async def close_modals(page: Page):
    for _ in range(3):
        try:
            btn = await page.query_selector(SEL_MODAL_CLOSE)
            if btn and await btn.is_visible():
                try: await page.click(SEL_MODAL_CLOSE, timeout=1500)
                except: await page.keyboard.press("Escape")
                await page.wait_for_timeout(300)
            else: return
        except Exception: return


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n>>> Прервано пользователем.")