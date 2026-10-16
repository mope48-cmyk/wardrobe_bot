# download_images.py
# Скачивает фото по ссылкам из файла links_to_check в папку clothes_images
# и создаёт CSV-заготовку со столбцами как в bot_test_results_with_urls.csv.
#
# Формат файла links_to_check:
#   https://example.com/photo.jpg
#   имя.jpg | https://example.com/photo.jpg
#   имя.jpg<TAB>https://example.com/photo.jpg
#
# Рекомендуемый формат имени файла:
#   NN_Категория_тип_цвет_материал_чек.png
#   (материал и чек можно опустить — тогда колонки останутся пустыми,
#    их можно дозаполнить вручную в CSV)
#
# Пустые строки и строки, начинающиеся с '#', игнорируются.
#
# Требует: pip install requests
#
# Запуск:
#   python download_images.py            # скачать + создать CSV
#   python download_images.py --no-csv   # только скачать, CSV не трогать

import csv
import hashlib
import sys
from pathlib import Path
from urllib.parse import urlparse, unquote

import requests

# === Конфигурация ===
LINKS_FILE = Path(r"C:\Projects\wardrobe_bot\tests\links_to_check.txt")
OUT_DIR    = Path(r"C:\Users\User\Desktop\Гардероб\clothes_images")
CSV_PATH   = Path(r"C:\Projects\wardrobe_bot\tests\bot_test_results_with_urls.csv")
TIMEOUT    = 60
USER_AGENT = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
              "AppleWebKit/537.36 (KHTML, like Gecko) "
              "Chrome/122.0 Safari/537.36")
OVERWRITE  = False   # True — перезаписывать уже скачанные файлы

NO_CSV = "--no-csv" in sys.argv

# Столбцы CSV — точно как в bot_test_results_with_urls.csv
CSV_FIELDS = [
    "num", "file", "source_url", "category", "type", "color",
    "material", "check", "sent_at", "response", "response_type",
    "response_time_sec",
]


# ---------- парсинг ссылок ----------

def filename_from_url(url: str) -> str:
    """Имя файла из URL."""
    path = unquote(urlparse(url).path)
    name = Path(path).name
    if not name or "." not in name:
        ext = ".png" if "png" in url.lower() else ".jpg"
        return hashlib.md5(url.encode()).hexdigest()[:12] + ext
    return name


def parse_line(line: str):
    """Из строки → (имя_файла, url) или None."""
    line = line.strip()
    if not line or line.startswith("#"):
        return None

    if "|" in line:
        a, b = (p.strip() for p in line.split("|", 1))
        return (filename_from_url(a), a) if a.startswith("http") else (a, b)

    if "\t" in line:
        a, b = (p.strip() for p in line.split("\t", 1))
        return (filename_from_url(a), a) if a.startswith("http") else (a, b)

    return filename_from_url(line), line


# ---------- метаданные из имени файла ----------

def parse_filename_meta(filename: str) -> dict:
    """
    Разбор имени файла на метаданные.

    Формат A (6+ частей):
        NN_Категория_тип_цвет_материал_чек...png
    Формат B (5 частей, без материала):
        NN_Категория_тип_цвет_чек...png
    Формат C (4 части):
        NN_Категория_тип_цвет.png

    Всё, что не удалось определить — пустая строка.
    """
    stem = Path(filename).stem
    parts = stem.split("_")
    meta = {k: "" for k in ("num", "category", "type", "color",
                            "material", "check")}

    if len(parts) >= 6:
        meta["num"]      = parts[0]
        meta["category"] = parts[1]
        meta["type"]     = parts[2]
        meta["color"]    = parts[3]
        meta["material"] = parts[4]
        meta["check"]    = "_".join(parts[5:])
    elif len(parts) == 5:
        meta["num"]      = parts[0]
        meta["category"] = parts[1]
        meta["type"]     = parts[2]
        meta["color"]    = parts[3]
        meta["check"]    = "_".join(parts[4:])
    elif len(parts) >= 4:
        meta["num"]      = parts[0]
        meta["category"] = parts[1]
        meta["type"]     = parts[2]
        meta["color"]    = parts[3]

    return meta


# ---------- CSV ----------

def load_existing_results() -> dict:
    """Читает существующий CSV → {filename: row_dict}."""
    results = {}
    if not CSV_PATH.exists():
        return results
    try:
        with open(CSV_PATH, encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                if row.get("file"):
                    results[row["file"]] = row
        print(f"ℹ Прочитано существующих строк CSV: {len(results)}")
    except Exception as e:
        print(f"⚠ Не удалось прочитать существующий CSV: {e}")
    return results


def write_results_csv(tasks: list, existing: dict) -> None:
    """
    Создаёт/обновляет CSV со структурой как bot_test_results_with_urls.csv.

    - Метаданные (source_url, category, type, color, material, check)
      берутся из существующего CSV, если там есть; иначе из имени файла
      и URL.
    - Результаты (sent_at, response, response_type, response_time_sec)
      сохраняются из существующего CSV, если они там были. Их заполнит
      autotest_browser.py.
    """
    rows = []
    for task in tasks:
        filename = task["filename"]
        url = task["url"]
        meta_name = parse_filename_meta(filename)
        old = existing.get(filename, {})

        row = {
            "num":        old.get("num")        or meta_name["num"],
            "file":       filename,
            "source_url": old.get("source_url") or url,
            "category":   old.get("category")   or meta_name["category"],
            "type":       old.get("type")       or meta_name["type"],
            "color":      old.get("color")      or meta_name["color"],
            "material":   old.get("material")   or meta_name["material"],
            "check":      old.get("check")      or meta_name["check"],
            "sent_at":          old.get("sent_at", ""),
            "response":         old.get("response", ""),
            "response_type":    old.get("response_type", ""),
            "response_time_sec": old.get("response_time_sec", ""),
        }
        rows.append(row)

    try:
        with open(CSV_PATH, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=CSV_FIELDS, quoting=csv.QUOTE_ALL
            )
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
        print(f"\n📄 CSV создан: {CSV_PATH}")
        print(f"   Строк: {len(rows)}")
    except Exception as e:
        print(f"⚠ Ошибка записи CSV: {e}")


# ---------- скачивание ----------

def download(session: requests.Session, url: str, dest: Path) -> bool:
    try:
        r = session.get(url, stream=True, timeout=TIMEOUT, allow_redirects=True)
        r.raise_for_status()

        ct = r.headers.get("Content-Type", "")
        if not (ct.startswith("image/") or ct.startswith("application/octet-stream")):
            print(f"    ⚠ Content-Type={ct!r} — возможно, это не картинка")

        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=65536):
                if chunk:
                    f.write(chunk)
        return True
    except Exception as e:
        print(f"    ❌ {e}")
        if dest.exists():
            dest.unlink()
        return False


# ---------- main ----------

def main():
    if not LINKS_FILE.exists():
        print(f"❌ Файл со ссылками не найден: {LINKS_FILE}")
        print("   Поправь LINKS_FILE в начале скрипта.")
        sys.exit(1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    lines = LINKS_FILE.read_text(encoding="utf-8").splitlines()
    tasks = [t for t in (parse_line(l) for l in lines) if t]

    if not tasks:
        print("❌ В файле нет валидных ссылок")
        sys.exit(1)

    # Загружаем результаты прошлого прогона, чтобы не потерять ответы бота
    existing = load_existing_results()

    print(f"Файл:        {LINKS_FILE}")
    print(f"Куда:        {OUT_DIR}")
    print(f"CSV:         {CSV_PATH}")
    print(f"Ссылок:      {len(tasks)}")
    print(f"Перезапись:  {'да' if OVERWRITE else 'нет (существующие пропускаются)'}")
    print()

    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT

    downloaded_tasks = []   # задачи, у которых файл точно есть на диске
    ok = skip = fail = 0

    for i, (name, url) in enumerate(tasks, 1):
        dest = OUT_DIR / name
        if dest.exists() and not OVERWRITE:
            print(f"[{i}/{len(tasks)}] ⏭ уже есть: {name}")
            skip += 1
            downloaded_tasks.append({"filename": name, "url": url})
            continue

        print(f"[{i}/{len(tasks)}] ↓ {name}")
        if download(session, url, dest):
            size = dest.stat().st_size
            print(f"    ✅ {size / 1024:.1f} KB")
            ok += 1
            downloaded_tasks.append({"filename": name, "url": url})
        else:
            fail += 1
            # файл не скачался — не включаем в CSV, чтобы не сбивать нумерацию

    print("\n=== ИТОГО ===")
    print(f"Скачано:   {ok}")
    print(f"Пропущено: {skip}")
    print(f"Ошибок:    {fail}")

    # Генерируем CSV, если не отключено
    if not NO_CSV:
        # Отдельно переносим результаты из existing в новый набор:
        # для строк, которых больше нет в links_to_check, оставляем как есть
        existing_for_removed = {
            k: v for k, v in existing.items()
            if k not in {t["filename"] for t in downloaded_tasks}
        }
        # Сохраняем их в отдельный файл на всякий случай
        if existing_for_removed:
            backup = CSV_PATH.with_suffix(".removed.csv")
            try:
                with open(backup, "w", encoding="utf-8", newline="") as f:
                    writer = csv.DictWriter(
                        f, fieldnames=CSV_FIELDS, quoting=csv.QUOTE_ALL
                    )
                    writer.writeheader()
                    for row in existing_for_removed.values():
                        writer.writerow({k: row.get(k, "") for k in CSV_FIELDS})
                print(f"ℹ {len(existing_for_removed)} строк без файла сохранены в {backup.name}")
            except Exception as e:
                print(f"⚠ Не удалось сохранить backup: {e}")

        write_results_csv(downloaded_tasks, existing)
    else:
        print("ℹ CSV не обновлялся (--no-csv)")


if __name__ == "__main__":
    main()
