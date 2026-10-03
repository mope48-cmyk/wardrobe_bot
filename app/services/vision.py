"""Распознавание одежды и определение цвета.

Категория и тип — через локальную модель CLIP (английские метки).
Цвет — через анализ пикселей (быстро и точно).
"""

import logging
from collections import Counter
from io import BytesIO

from PIL import Image
from transformers import pipeline

logger = logging.getLogger(__name__)

MODEL_ID = "openai/clip-vit-base-patch32"

# Английские метки для CLIP (он обучался на английском).
ENGLISH_LABELS = [
    # Верх
    "a jacket", "a coat", "a raincoat", "a sweater", "a hoodie",
    "a cardigan", "a shirt", "a t-shirt", "a blouse", "a top",
    "a dress",
    # Низ
    "trousers", "jeans", "shorts", "a skirt", "leggings",
    # Обувь
    "sneakers", "boots", "shoes", "sandals",
    # Аксессуары
    "a hat", "a scarf", "gloves", "a belt", "a bag",
]

# Английская метка → (русский тип, категория)
LABEL_MAP = {
    "a jacket": ("куртка", "верх"),
    "a coat": ("пальто", "верх"),
    "a raincoat": ("плащ", "верх"),
    "a sweater": ("свитер", "верх"),
    "a hoodie": ("толстовка", "верх"),
    "a cardigan": ("кардиган", "верх"),
    "a shirt": ("рубашка", "верх"),
    "a t-shirt": ("футболка", "верх"),
    "a blouse": ("блузка", "верх"),
    "a top": ("топ", "верх"),
    "a dress": ("платье", "верх"),
    "trousers": ("брюки", "низ"),
    "jeans": ("джинсы", "низ"),
    "shorts": ("шорты", "низ"),
    "a skirt": ("юбка", "низ"),
    "leggings": ("леггинсы", "низ"),
    "sneakers": ("кроссовки", "обувь"),
    "boots": ("ботинки", "обувь"),
    "shoes": ("туфли", "обувь"),
    "sandals": ("сандалии", "обувь"),
    "a hat": ("шапка", "аксессуар"),
    "a scarf": ("шарф", "аксессуар"),
    "gloves": ("перчатки", "аксессуар"),
    "a belt": ("ремень", "аксессуар"),
    "a bag": ("сумка", "аксессуар"),
}

# Палитра цветов. RGB-эталоны подобраны так, чтобы охватить типичные цвета одежды.
COLOR_PALETTE = {
    "чёрный":      (30, 30, 30),
    "белый":       (245, 245, 245),
    "серый":       (128, 128, 128),
    "бежевый":     (222, 202, 165),
    "коричневый":  (110, 70, 40),
    "красный":     (200, 40, 40),
    "бордовый":    (110, 20, 40),
    "оранжевый":   (240, 130, 30),
    "жёлтый":      (240, 220, 60),
    "зелёный":     (50, 150, 60),
    "голубой":     (120, 180, 230),
    "синий":       (40, 70, 180),
    "фиолетовый":  (130, 70, 180),
    "розовый":     (240, 140, 180),
}


def _nearest_color(rgb: tuple[int, int, int]) -> str:
    """Найти ближайший именованный цвет по евклидову расстоянию в RGB."""
    r, g, b = rgb
    best_name = "неизвестный"
    best_dist = float("inf")
    for name, (cr, cg, cb) in COLOR_PALETTE.items():
        d = (r - cr) ** 2 + (g - cg) ** 2 + (b - cb) ** 2
        if d < best_dist:
            best_dist = d
            best_name = name
    return best_name


def detect_color(image_bytes: bytes) -> tuple[str, list[str]]:
    """Определить доминирующий цвет и топ-3 вероятных.

    Возвращает (primary_color, [top_colors]).
    """
    img = Image.open(BytesIO(image_bytes)).convert("RGB")
    # Уменьшаем для скорости — цвет не требует деталей.
    img.thumbnail((150, 150))
    pixels = list(img.getdata())

    counter = Counter()
    for p in pixels:
        counter[_nearest_color(p)] += 1

    if not counter:
        return "неизвестный", []

    top = counter.most_common(3)
    primary = top[0][0]
    return primary, [name for name, _ in top]


# ---------- CLIP ----------

_classifier = None


def _get_classifier():
    global _classifier
    if _classifier is None:
        logger.info("Загружаем модель %s...", MODEL_ID)
        _classifier = pipeline(
            task="zero-shot-image-classification",
            model=MODEL_ID,
            device=-1,
        )
        logger.info("Модель загружена")
    return _classifier


async def classify_clothing(image_bytes: bytes) -> dict:
    """Распознать категорию и тип вещи. Также определяет цвет.

    Возвращает:
    {
        "ok": bool,
        "category": "верх" | "низ" | "обувь" | "аксессуар" | "другое",
        "type": "футболка" | ...,
        "confidence": float,
        "color": "белый" | ...,
        "color_candidates": ["белый", "серый", "бежевый"],
    }
    """
    # Цвет определяем всегда, независимо от CLIP
    color_primary, color_candidates = detect_color(image_bytes)

    try:
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        classifier = _get_classifier()
        results = classifier(image, candidate_labels=ENGLISH_LABELS)
    except Exception:
        logger.exception("Ошибка классификации")
        return {
            "ok": False,
            "color": color_primary,
            "color_candidates": color_candidates,
        }

    if not results:
        return {
            "ok": False,
            "color": color_primary,
            "color_candidates": color_candidates,
        }

    top = results[0]
    en_label = top["label"]
    confidence = float(top["score"])

    ru_type, category = LABEL_MAP.get(en_label, (en_label, "другое"))

    logger.info(
        "Распознавание: en=%s ru=%s category=%s confidence=%.3f color=%s",
        en_label, ru_type, category, confidence, color_primary,
    )

    return {
        "ok": True,
        "category": category,
        "type": ru_type,
        "confidence": confidence,
        "color": color_primary,
        "color_candidates": color_candidates,
    }