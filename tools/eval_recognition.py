"""Оценка распознавания одежды на датасете с метками в именах файлов.

Имя файла:  NN_Категория_Тип_Цвет[_доп].jpg
Пример:      05_Верх_рубашка_синий_на_белом.jpg

Запуски:
    # 1. Локально, без моделей — только проверить парсинг имён:
    python tools/eval_recognition.py <папка> --dry-run

    # 2. Полный прогон на сервере (там стоят модели) внутри контейнера:
    docker compose exec bot python tools/eval_recognition.py /app/dataset_clothes_images
"""

import argparse
import asyncio
import csv
import logging
import re
import sys
from pathlib import Path

# Корень проекта (родитель tools/) — чтобы работал `import app.*`
# при запуске как `python tools/eval_recognition.py`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("eval_recognition")

CAT_MAP = {
    "верх": "верх",
    "низ": "низ",
    "обувь": "обувь",
    "аксессуары": "аксессуар",
    "аксессуар": "аксессуар",
}

# Синонимичные группы типов: совпадение внутри группы считается верным.
TYPE_ALIASES = {
    "толстовка": "толстовка/свитшот/худи",
    "свитшот": "толстовка/свитшот/худи",
    "худи": "толстовка/свитшот/худи",
    "зип худи": "толстовка/свитшот/худи",
    "толстовка с капюшоном": "толстовка/свитшот/худи",
    "пиджак": "пиджак/блейзер",
    "блейзер": "пиджак/блейзер",
    "ботинки": "ботинки/ботильоны",
    "ботильоны": "ботинки/ботильоны",
    "куртка-бомбер": "бомбер",
    "бомбер": "бомбер",
    "рубашка": "рубашка/лонгслив",
    "лонгслив": "рубашка/лонгслив",
    "спортивные штаны": "спортивные штаны/джоггеры",
    "джоггеры": "спортивные штаны/джоггеры",
    "свитер": "свитер/шерстяной свитер/вязаный свитер",
    "шерстяной свитер": "свитер/шерстяной свитер/вязаный свитер",
    "вязаный свитер": "свитер/шерстяной свитер/вязаный свитер",
}

# Канонические формы цветов: разные написания одной сущности.
# Ключи — уже нормализованные (без ё, нижний регистр).
CANON_COLOR_MAP = {
    "черный": "чёрный",
    "темно-синий": "тёмно-синий",
    "голубые": "голубой",
    "голубая": "голубой",
    "светло-серый": "светло-серый",
    "тёмно-коричневый": "тёмно-коричневый",
}


def canon_color(s: str) -> str:
    return CANON_COLOR_MAP.get(_norm(s), _norm(s))


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower()).replace("ё", "е")


def parse_filename(name: str) -> dict | None:
    """Разобрать имя файла в {category, type, color}."""
    stem = Path(name).stem
    parts = [p for p in stem.split("_") if p]
    if len(parts) < 3:
        return None

    category = CAT_MAP.get(_norm(parts[1]))
    if not category:
        return None

    item_type = _norm(parts[2])
    color = None
    color_raw = parts[3].lower() if len(parts) >= 4 else ""
    if color_raw and not color_raw.startswith("не") and not color_raw.startswith("на"):
        color = color_raw

    return {"category": category, "type": item_type, "color": color}


def type_group(t: str) -> str:
    return TYPE_ALIASES.get(t, t)


def load_manifest(folder: Path) -> dict[str, dict] | None:
    """Если рядом есть manifest.csv — берём метки оттуда (полные).

    Колонки: file, category, type, color. Файлы в манифесте имеют
    приоритет над парсингом имён.
    """
    mf = folder / "manifest.csv"
    if not mf.exists():
        return None
    import csv as _csv

    with open(mf, newline="", encoding="utf-8") as f:
        rows = list(_csv.DictReader(f))
    return {
        r["file"]: {
            "category": (r["category"] or "").strip().lower(),
            "type": (r["type"] or "").strip(),
            "color": ((r.get("color") or "").strip() or None),
        }
        for r in rows
    }


def collect_labels(folder: Path) -> tuple[list[Path], list[dict]]:
    manifest = load_manifest(folder)
    files, labels = [], []
    for p in sorted(folder.iterdir()):
        if not p.is_file() or p.suffix.lower() not in (".jpg", ".jpeg", ".png", ".webp"):
            continue
        label = (manifest or {}).get(p.name)
        if label is None and manifest is not None:
            continue  # есть манифест — файл вне его не оцениваем
        if label is None:
            label = parse_filename(p.name)
            if label is None:
                logger.warning("Пропуск (не парсится): %s", p.name)
                continue
        files.append(p)
        labels.append(label)
    return files, labels


async def classify_one(path: Path) -> dict:
    from app.services.vision import classify_clothing

    data = await classify_clothing(path.read_bytes())
    return data


async def evaluate(folder: Path) -> dict:
    files, labels = collect_labels(folder)
    if not files:
        logger.error("Нет распознаваемых файлов в %s", folder)
        sys.exit(1)

    cat_correct = cat_total = 0
    type_correct = type_total = 0
    color_correct = color_total = 0
    color3_correct = 0
    rows = []
    cat_cm: dict[str, dict[str, int]] = {}
    type_cm: dict[str, dict[str, int]] = {}

    for path, label in zip(files, labels, strict=False):
        pred = await classify_one(path)
        diag = pred.get("diagnostics") or {}
        row = {
            "file": path.name,
            "gt_category": label["category"],
            "gt_type": label["type"],
            "gt_color": label["color"] or "",
            "pred_ok": pred.get("ok", False),
            "pred_category": pred.get("category", ""),
            "pred_type": pred.get("type", ""),
            "pred_color": pred.get("color", ""),
            "pred_detected": "|".join(pred.get("detected_colors") or []),
            "confidence": round(float(pred.get("confidence", 0.0)), 4),
            "mask_on": diag.get("mask_on", ""),
            "dark_ratio": diag.get("dark_ratio", ""),
            "colored_ratio": diag.get("colored_ratio", ""),
            "avg_v": diag.get("avg_v", ""),
        }
        rows.append(row)

        # Категория
        cat_total += 1
        hit = pred.get("ok") and pred.get("category") == label["category"]
        cat_correct += int(hit)
        cat_cm.setdefault(label["category"], {}).setdefault(
            pred.get("category") or "—", 0
        )
        cat_cm[label["category"]][pred.get("category") or "—"] += 1

        # Тип (с синонимичными группами)
        type_total += 1
        type_hit = (
            pred.get("ok")
            and type_group(pred.get("type", "")) == type_group(label["type"])
        )
        type_correct += int(type_hit)
        type_cm.setdefault(label["type"], {}).setdefault(
            pred.get("type") or "—", 0
        )
        type_cm[label["type"]][pred.get("type") or "—"] += 1

        # Цвет (только где есть gt): топ-1 (канонизир.) и топ-3
        if label["color"]:
            color_total += 1
            color_hit = canon_color(pred.get("color", "")) == canon_color(label["color"])
            color_correct += int(color_hit)

            detected = pred.get("detected_colors") or []
            top3_hit = any(
                canon_color(c) == canon_color(label["color"]) for c in detected
            )
            color3_correct += int(top3_hit)
            row["color_top3_hit"] = top3_hit
        else:
            row["color_top3_hit"] = ""

    return {
        "total": len(files),
        "cat_acc": cat_correct / cat_total if cat_total else 0.0,
        "type_acc": type_correct / type_total if type_total else 0.0,
        "color_acc": color_correct / color_total if color_total else 0.0,
        "color3_acc": color3_correct / color_total if color_total else 0.0,
        "color_labeled": color_total,
        "rows": rows,
        "cat_cm": cat_cm,
        "type_cm": type_cm,
    }


def print_report(res: dict) -> None:
    print("\n" + "=" * 60)
    print(f"Файлов: {res['total']}")
    print(f"Точность категории:  {res['cat_acc']*100:.1f}%")
    print(f"Точность типа:       {res['type_acc']*100:.1f}%")
    print(
        f"Точность цвета (топ-1): {res['color_acc']*100:.1f}% "
        f"(на {res['color_labeled']} фото с меткой цвета)"
    )
    print(
        f"Точность цвета (топ-3): {res['color3_acc']*100:.1f}% "
        f"(GT есть среди detected_colors)"
    )
    print("=" * 60)

    print("\nСамые частые ошибки категории (gt → предсказание):")
    counts: dict[tuple[str, str], int] = {}
    for gt, m in res["cat_cm"].items():
        for pred, n in m.items():
            if pred != gt:
                counts[(gt, pred)] = n
    for (gt, pred), n in sorted(counts.items(), key=lambda x: -x[1])[:8]:
        print(f"  {gt} → {pred}: {n}")

    print("\nСамые частые ошибки типа (gt → предсказание):")
    counts = {}
    for gt, m in res["type_cm"].items():
        for pred, n in m.items():
            if type_group(pred) != type_group(gt):
                counts[(gt, pred)] = n
    for (gt, pred), n in sorted(counts.items(), key=lambda x: -x[1])[:10]:
        print(f"  {gt} → {pred}: {n}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--dry-run", action="store_true",
                        help="только парсинг имён, без моделей")
    parser.add_argument("--csv", type=Path, default=None,
                        help="сохранить построчные результаты в CSV")
    args = parser.parse_args()

    if not args.folder.is_dir():
        logger.error("Папка не найдена: %s", args.folder)
        sys.exit(1)

    files, labels = collect_labels(args.folder)
    print(f"Разобрано имён: {len(files)}")
    print(
        "Файлов с меткой цвета: "
        f"{sum(1 for lbl in labels if lbl['color'])}"
    )

    if args.dry_run:
        print("\nПримеры распарсенных меток:")
        for p, lbl in zip(files[:10], labels[:10], strict=False):
            print(f"  {p.name}: {lbl}")
        print("\n--dry-run завершён (модели не запускались).")
        return

    res = asyncio.run(evaluate(args.folder))
    print_report(res)

    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(res["rows"][0].keys()))
            writer.writeheader()
            writer.writerows(res["rows"])
        print(f"\nПострочные результаты: {args.csv}")


if __name__ == "__main__":
    main()
