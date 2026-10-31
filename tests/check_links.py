# check_links.py
# Проверка прямых ссылок на изображения для тестового набора.
# Запуск: python check_links.py

import asyncio
import csv
from pathlib import Path
from playwright.async_api import async_playwright

CSV_PATH = Path(r"C:\Projects\wardrobe_bot\tests\bot_test_results_with_urls.csv")
LINKS_FILE = Path(r"C:\Projects\wardrobe_bot\tests\links_to_check.txt")


async def check_link(page, url: str) -> dict:
    """Открывает ссылку и возвращает результат проверки."""
    result = {"url": url, "status": "", "is_image": False, "error": ""}
    try:
        response = await page.goto(url, wait_until="domcontentloaded", timeout=15000)
        result["status"] = response.status if response else "no response"

        # Проверяем content-type
        content_type = ""
        if response:
            content_type = response.headers.get("content-type", "")
        result["is_image"] = "image" in content_type

        # Проверяем, что страница не пустая
        if not result["is_image"]:
            body = await page.inner_text("body")
            result["error"] = f"content-type={content_type}, body_len={len(body)}"

    except Exception as e:
        result["status"] = "ERROR"
        result["error"] = str(e)[:100]

    return result


async def main():
    if not LINKS_FILE.exists():
        print(f"❌ Файл со ссылками не найден: {LINKS_FILE}")
        print("Создайте файл links_to_check.txt и вставьте по одной ссылке в строке.")
        return

    urls = [
        line.strip() for line in LINKS_FILE.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]
    print(f">>> Проверяю {len(urls)} ссылок...\n")

    results = []
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context()
        page = await context.new_page()

        for i, url in enumerate(urls, 1):
            res = await check_link(page, url)
            icon = "✅" if res["is_image"] else "❌"
            print(f"{icon} [{i}/{len(urls)}] {url[:80]}")
            if not res["is_image"]:
                print(f"      → {res['status']} | {res['error'][:80]}")
            results.append(res)
            await asyncio.sleep(0.3)

        await browser.close()

    # Сохраняем отчёт
    ok = sum(1 for r in results if r["is_image"])
    print(f"\n=== ИТОГО ===")
    print(f"Всего ссылок:     {len(results)}")
    print(f"Рабочих (image):  {ok}")
    print(f"Проблемных:       {len(results) - ok}")

    # Записываем проблемные
    bad = [r for r in results if not r["is_image"]]
    if bad:
        bad_file = Path("bad_links.txt")
        with open(bad_file, "w", encoding="utf-8") as f:
            for r in bad:
                f.write(f"{r['url']}\n# {r['status']} | {r['error']}\n\n")
        print(f"\n>>> Проблемные ссылки записаны в: {bad_file}")


if __name__ == "__main__":
    asyncio.run(main())
