"""Распознавание одежды и определение цвета."""

import logging
from collections import Counter
from io import BytesIO

from PIL import Image
from transformers import pipeline

logger = logging.getLogger(__name__)

MODEL_ID = "openai/clip-vit-base-patch32"

# Расширенный список английских меток.
ENGLISH_LABELS = [
    # Верх
    "a jacket", "a coat", "a raincoat", "a windbreaker", "a down jacket",
    "a leather jacket", "a denim jacket", "a sweater", "a hoodie",
    "a sweatshirt", "a cardigan", "a shirt", "a t-shirt", "a polo shirt",
    "a blouse", "a top", "a dress", "a suit", "a vest",
    # Низ
    "trousers", "jeans", "shorts", "a skirt", "leggings", "sweatpants",
    # Обувь
    "sneakers", "boots", "shoes", "sandals", "heels", "loafers",
    # Аксессуары
    "a hat", "a cap", "a scarf", "gloves", "a belt", "a bag", "a backpack",
]

# Английская метка → (русский тип, категория)
LABEL_MAP = {
    "a jacket": ("куртка", "верх"),
    "a coat": ("пальто", "верх"),
    "a raincoat": ("плащ", "верх"),
    "a windbreaker": ("ветровка", "верх"),
    "a down jacket": ("пуховик", "верх"),
    "a leather jacket": ("кожаная куртка", "верх"),
    "a denim jacket": ("джинсовая куртка", "верх"),
    "a sweater": ("свитер", "верх"),
    "a hoodie": ("толстовка", "верх"),
    "a sweatshirt": ("свитшот", "верх"),
    "a cardigan": ("кардиган", "верх"),
    "a shirt": ("рубашка", "верх"),
    "a t-shirt": ("футболка", "верх"),
    "a polo shirt": ("поло", "верх"),
    "a blouse": ("блузка", "верх"),
    "a top": ("топ", "верх"),
    "a dress": ("платье", "верх"),
    "a suit": ("костюм", "верх"),
    "a vest": ("жилет", "верх"),
    "trousers": ("брюки", "низ"),
    "jeans": ("джинсы", "низ"),
    "shorts": ("шорты", "низ"),
    "a skirt": ("юбка", "низ"),
    "leggings": ("леггинсы", "низ"),
    "sweatpants": ("спортивные штаны", "низ"),
    "sneakers": ("кроссовки", "обувь"),
    "boots": ("ботинки", "обувь"),
    "shoes": ("туфли", "обувь"),
    "sandals": ("сандалии", "обувь"),
    "heels": ("каблуки", "обувь"),
    "loafers": ("лоферы", "обувь"),
    "a hat": ("шапка", "аксессуар"),
    "a cap": ("кепка", "аксессуар"),
    "a scarf": ("шарф", "аксессуар"),
    "gloves": ("перчатки", "аксессуар"),
    "a belt": ("ремень", "аксессуар"),
    "a bag": ("сумка", "аксессуар"),
    "a backpack": ("рюкзак", "аксессуар"),
}

# Дефолтные характеристики по русскому типу вещи.
# Формат: тип → (warmth_level, waterproof, formal_level, season)
TYPE_DEFAULTS = {
    # Верх
    "куртка":            (4, False, "casual", "демисезон"),
    "пальто":            (4, False, "business", "зима"),
    "плащ":              (3, True,  "casual", "демисезон"),
    "ветровка":          (2, True,  "casual", "демисезон"),
    "пуховик":           (5, True,  "casual", "зима"),
    "кожаная куртка":    (3, True,  "casual", "демисезон"),
    "джинсовая куртка":  (3, False, "casual", "демисезон"),
    "свитер":            (4, False, "casual", "зима"),
    "толстовка":         (3, False, "casual", "демисезон"),
    "свитшот":           (3, False, "casual", "демисезон"),
    "кардиган":          (3, False, "casual", "демисезон"),
    "рубашка":           (2, False, "business", "универсальная"),
    "футболка":          (1, False, "casual", "лето"),
    "поло":              (2, False, "casual", "лето"),
    "блузка":            (2, False, "business", "универсальная"),
    "топ":               (1, False, "casual", "лето"),
    "платье":            (2, False, "business", "лето"),
    "костюм":            (3, False, "business", "универсальная"),
    "жилет":             (2, False, "casual", "демисезон"),
    # Низ
    "брюки":             (3, False, "business", "универсальная"),
    "джинсы":            (3, False, "casual", "универсальная"),
    "шорты":             (1, False, "casual", "лето"),
    "юбка":              (2, False, "casual", "лето"),
    "леггинсы":          (2, False, "sport", "демисезон"),
    "спортивные штаны":  (2, False, "sport", "демисезон"),
    # Обувь
    "кроссовки":         (2, False, "sport", "демисезон"),
    "ботинки":           (3, True,  "casual", "демисезон"),
    "туфли":             (2, False, "business", "универсальная"),
    "сандалии":          (1, False, "casual", "лето"),
    "каблуки":           (2, False, "business", "универсальная"),
    "лоферы":            (2, False, "business", "универсальная"),
    # Аксессуары
    "шапка":             (4, False, "casual", "зима"),
    "кепка":             (1, False, "casual", "лето"),
    "шарф":              (4, False, "casual", "зима"),
    "перчатки":          (3, False, "casual", "зима"),
    "ремень":            (1, False, "business", "универсальная"),
    "сумка":             (1, False, "casual", "универсальная"),
    "рюкзак":            (1, False, "casual", "универсальная"),
}

# ---------- Определение цвета ----------

import colorsys


def _rgb_to_color_name(r: int, g: int, b: int) -> str:
    """Определить имя цвета по RGB через HSV.

    HSV отделяет оттенок (Hue) от яркости (Value) и насыщенности (Saturation),
    поэтому тени и освещение влияют слабее, чем в чистом RGB.
    """
    h, s, v = colorsys.rgb_to_hsv(r / 255, g / 255, b / 255)
    h_deg = h * 360      # 0–360: тон
    s_pct = s * 100      # 0–100: насыщенность
    v_pct = v * 100      # 0–100: яркость

    # Ахроматические цвета (низкая насыщенность)
    if s_pct < 15:
        if v_pct < 20:
            return "чёрный"
        if v_pct > 85:
            return "белый"
        return "серый"

    # Коричневый — тёмный и умеренно насыщенный, в оранжево-жёлтом диапазоне
    if 15 <= h_deg < 50 and v_pct < 55 and s_pct > 20:
        return "коричневый"

    # Бежевый — светлый, слабонасыщенный, тёплый
    if 20 <= h_deg < 55 and s_pct < 40 and v_pct > 60:
        return "бежевый"

    # Цветные по тону
    if h_deg < 15 or h_deg >= 345:
        if v_pct < 50:
            return "бордовый"
        return "красный"
    if h_deg < 40:
        return "оранжевый"
    if h_deg < 70:
        return "жёлтый"
    if h_deg < 165:
        return "зелёный"
    if h_deg < 200:
        return "голубой"
    if h_deg < 255:
        return "синий"
    if h_deg < 290:
        return "фиолетовый"
    if h_deg < 345:
        return "розовый"

    return "неизвестный"


def detect_color(image_bytes: bytes) -> tuple[str, list[str]]:
    """Определить доминирующий цвет и топ-3.

    Алгоритм:
    1. Уменьшаем изображение до 200×200 (для скорости).
    2. Отрезаем 15% по краям — вещь почти всегда в центре, фон по краям.
    3. Квантуем в 5 доминирующих цветов (встроенный PIL quantize).
    4. Каждый кластер переводим в имя цвета через HSV.
    5. Возвращаем самый частый как основной + топ-3 уникальных имён.
    """
    img = Image.open(BytesIO(image_bytes)).convert("RGB")
    img.thumbnail((200, 200))

    w, h = img.size
    margin_x = int(w * 0.15)
    margin_y = int(h * 0.15)
    img = img.crop((margin_x, margin_y, w - margin_x, h - margin_y))

    # Квантование: 5 доминирующих цветов на изображении.
    # MEDIANCUT — быстрый и хорошо работает на фотографиях.
    quantized = img.quantize(colors=5, method=Image.Quantize.MEDIANCUT)
    palette = quantized.getpalette() or []
    counts = quantized.getcolors(maxcolors=256) or []

    if not counts:
        return "неизвестный", []

    # counts = [(count, palette_index), ...]. Сортируем по частоте.
    counts.sort(key=lambda x: x[0], reverse=True)

    color_names: list[str] = []
    for _, idx in counts:
        r = palette[idx * 3]
        g = palette[idx * 3 + 1]
        b = palette[idx * 3 + 2]
        name = _rgb_to_color_name(r, g, b)
        if name not in color_names:
            color_names.append(name)

    primary = color_names[0] if color_names else "неизвестный"
    return primary, color_names[:3]

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
    """Распознать вещь и вернуть все атрибуты, включая дефолтные.

    Возвращает:
    {
        "ok": bool,
        "category": "верх" | "низ" | "обувь" | "аксессуар" | "другое",
        "type": "футболка" | ...,
        "confidence": float,
        "color": "белый" | ...,
        "color_candidates": [...],
        "warmth_level": int,
        "waterproof": bool,
        "formal_level": str,
        "season": str,
    }
    """
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

    # Дефолтные характеристики по типу
    defaults = TYPE_DEFAULTS.get(ru_type, (3, False, "casual", "универсальная"))
    warmth, waterproof, formal, season = defaults

    logger.info(
        "Распознавание: en=%s ru=%s category=%s confidence=%.3f "
        "color=%s warmth=%s waterproof=%s formal=%s season=%s",
        en_label, ru_type, category, confidence,
        color_primary, warmth, waterproof, formal, season,
    )

    return {
        "ok": True,
        "category": category,
        "type": ru_type,
        "confidence": confidence,
        "color": color_primary,
        "color_candidates": color_candidates,
        "warmth_level": warmth,
        "waterproof": waterproof,
        "formal_level": formal,
        "season": season,
    }