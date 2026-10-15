"""Распознавание одежды через SigLIP. Цвет — HSV-анализ.

Florence-2 убран из основного пайплайна — слишком медленный на CPU.
Материал пользователь указывает кнопкой после распознавания типа.

Модели загружаются лениво, прогреваются при старте бота.
"""

import colorsys
import hashlib
import logging
import os
from io import BytesIO

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

from app.services.segmentation import crop_garment, warmup_segmentation

torch.set_num_threads(4)

logger = logging.getLogger(__name__)

SIGLIP_MODEL_ID = "google/siglip-so400m-patch14-384"
SIGLIP_THRESHOLD = 0.15

_SIGLIP_CACHE_MAX = 128
_siglip_cache: dict[str, list[dict]] = {}
_siglip_cache_order: list[str] = []


def _siglip_cache_get(image_bytes: bytes) -> list[dict] | None:
    key = hashlib.md5(image_bytes).hexdigest()
    if key in _siglip_cache:
        logger.debug("SigLIP cache hit: %s", key[:8])
        return _siglip_cache[key]
    return None


def _siglip_cache_set(image_bytes: bytes, result: list[dict]) -> None:
    key = hashlib.md5(image_bytes).hexdigest()
    if key in _siglip_cache:
        return
    _siglip_cache[key] = result
    _siglip_cache_order.append(key)
    if len(_siglip_cache_order) > _SIGLIP_CACHE_MAX:
        old_key = _siglip_cache_order.pop(0)
        _siglip_cache.pop(old_key, None)


# ---------- Метки ----------

ENGLISH_LABELS = [
    "a jacket", "a winter jacket",
    "a coat", "a long coat",
    "a raincoat", "a waterproof jacket",
    "a windbreaker",
    "a puffer jacket",
    "a leather jacket",
    "a denim jacket",
    "a suit jacket", "a blazer",
    "a bomber jacket", "a parka", "a trench coat",
    "a sweater", "a knitted sweater", "a wool sweater",
    "a hoodie", "a sweatshirt", "a cardigan",
    "a vest", "a turtleneck", "a long-sleeve shirt",
    "a shirt", "a t-shirt", "a polo shirt",
    "a blouse", "a top", "a tank top",
    "a dress",
    "trousers", "jeans", "shorts",
    "a skirt", "leggings", "sweatpants", "joggers",
    "sneakers", "high-top sneakers", "running shoes",
    "boots", "ankle boots", "winter boots",
    "shoes", "loafers", "ballet flats", "slippers",
    "sandals", "heels",
    "a hat", "a beanie", "a cap", "a scarf", "gloves",
    "a belt", "a bag", "a backpack",
    "a landscape", "a room interior", "furniture",
    "food", "an animal", "a person's face",
    "a text document", "an empty background",
]

LABEL_MAP = {
    "a jacket": ("куртка", "верх"),
    "a winter jacket": ("зимняя куртка", "верх"),
    "a coat": ("пальто", "верх"),
    "a long coat": ("длинное пальто", "верх"),
    "a raincoat": ("дождевик", "верх"),
    "a waterproof jacket": ("непромокаемая куртка", "верх"),
    "a windbreaker": ("ветровка", "верх"),
    "a puffer jacket": ("пуховик", "верх"),
    "a leather jacket": ("кожаная куртка", "верх"),
    "a denim jacket": ("джинсовая куртка", "верх"),
    "a suit jacket": ("пиджак", "верх"),
    "a blazer": ("блейзер", "верх"),
    "a bomber jacket": ("бомбер", "верх"),
    "a parka": ("парка", "верх"),
    "a trench coat": ("тренч", "верх"),
    "a sweater": ("свитер", "верх"),
    "a knitted sweater": ("вязаный свитер", "верх"),
    "a wool sweater": ("шерстяной свитер", "верх"),
    "a hoodie": ("толстовка", "верх"),
    "a sweatshirt": ("свитшот", "верх"),
    "a cardigan": ("кардиган", "верх"),
    "a vest": ("жилет", "верх"),
    "a turtleneck": ("водолазка", "верх"),
    "a long-sleeve shirt": ("лонгслив", "верх"),
    "a shirt": ("рубашка", "верх"),
    "a t-shirt": ("футболка", "верх"),
    "a polo shirt": ("поло", "верх"),
    "a blouse": ("блузка", "верх"),
    "a top": ("топ", "верх"),
    "a tank top": ("майка", "верх"),
    "a dress": ("платье", "верх"),
    "trousers": ("брюки", "низ"),
    "jeans": ("джинсы", "низ"),
    "shorts": ("шорты", "низ"),
    "a skirt": ("юбка", "низ"),
    "leggings": ("леггинсы", "низ"),
    "sweatpants": ("спортивные штаны", "низ"),
    "joggers": ("джоггеры", "низ"),
    "sneakers": ("кроссовки", "обувь"),
    "high-top sneakers": ("высокие кроссовки", "обувь"),
    "running shoes": ("беговые кроссовки", "обувь"),
    "boots": ("ботинки", "обувь"),
    "ankle boots": ("ботильоны", "обувь"),
    "winter boots": ("зимние ботинки", "обувь"),
    "shoes": ("туфли", "обувь"),
    "loafers": ("лоферы", "обувь"),
    "ballet flats": ("балетки", "обувь"),
    "slippers": ("тапочки", "обувь"),
    "sandals": ("сандалии", "обувь"),
    "heels": ("каблуки", "обувь"),
    "a hat": ("шляпа", "аксессуар"),
    "a beanie": ("шапка-бини", "аксессуар"),
    "a cap": ("кепка", "аксессуар"),
    "a scarf": ("шарф", "аксессуар"),
    "gloves": ("перчатки", "аксессуар"),
    "a belt": ("ремень", "аксессуар"),
    "a bag": ("сумка", "аксессуар"),
    "a backpack": ("рюкзак", "аксессуар"),
}

NEGATIVE_LABELS = {
    "a landscape", "a room interior", "furniture",
    "food", "an animal", "a person's face",
    "a text document", "an empty background",
}

TYPE_DEFAULTS = {
    "куртка":              (4, False, "casual", "демисезон"),
    "зимняя куртка":       (5, True,  "casual", "зима"),
    "пальто":              (4, False, "business", "зима"),
    "длинное пальто":      (4, False, "business", "зима"),
    "дождевик":            (2, True,  "casual", "демисезон"),
    "непромокаемая куртка": (4, True, "casual", "демисезон"),
    "ветровка":            (2, True,  "casual", "демисезон"),
    "пуховик":             (5, True,  "casual", "зима"),
    "кожаная куртка":      (3, True,  "casual", "демисезон"),
    "джинсовая куртка":    (3, False, "casual", "демисезон"),
    "пиджак":              (3, False, "business", "универсальная"),
    "блейзер":             (3, False, "business", "универсальная"),
    "бомбер":              (3, False, "casual", "демисезон"),
    "парка":               (5, True,  "casual", "зима"),
    "тренч":               (4, True,  "business", "демисезон"),
    "свитер":              (4, False, "casual", "зима"),
    "вязаный свитер":      (4, False, "casual", "зима"),
    "шерстяной свитер":    (5, False, "casual", "зима"),
    "толстовка":           (3, False, "casual", "демисезон"),
    "свитшот":             (3, False, "casual", "демисезон"),
    "кардиган":            (3, False, "casual", "демисезон"),
    "жилет":               (2, False, "casual", "демисезон"),
    "водолазка":           (3, False, "casual", "демисезон"),
    "лонгслив":            (2, False, "casual", "демисезон"),
    "рубашка":             (2, False, "business", "универсальная"),
    "футболка":            (1, False, "casual", "лето"),
    "поло":                (2, False, "casual", "лето"),
    "блузка":              (2, False, "business", "универсальная"),
    "топ":                 (1, False, "casual", "лето"),
    "майка":               (1, False, "casual", "лето"),
    "платье":              (2, False, "business", "лето"),
    "брюки":               (3, False, "business", "универсальная"),
    "джинсы":              (3, False, "casual", "универсальная"),
    "шорты":               (1, False, "casual", "лето"),
    "юбка":                (2, False, "casual", "лето"),
    "леггинсы":            (2, False, "sport", "демисезон"),
    "спортивные штаны":    (2, False, "sport", "демисезон"),
    "джоггеры":            (2, False, "sport", "демисезон"),
    "кроссовки":           (2, False, "sport", "демисезон"),
    "высокие кроссовки":   (2, False, "sport", "демисезон"),
    "беговые кроссовки":   (2, False, "sport", "лето"),
    "ботинки":             (3, True,  "casual", "демисезон"),
    "ботильоны":           (3, False, "casual", "демисезон"),
    "зимние ботинки":      (4, True,  "casual", "зима"),
    "туфли":               (2, False, "business", "универсальная"),
    "лоферы":              (2, False, "business", "универсальная"),
    "балетки":             (1, False, "casual", "лето"),
    "тапочки":             (1, False, "casual", "универсальная"),
    "сандалии":            (1, False, "casual", "лето"),
    "каблуки":             (2, False, "business", "универсальная"),
    "шляпа":               (2, False, "casual", "лето"),
    "шапка-бини":          (4, False, "casual", "зима"),
    "кепка":               (1, False, "casual", "лето"),
    "шарф":                (4, False, "casual", "зима"),
    "перчатки":            (3, False, "casual", "зима"),
    "ремень":              (1, False, "business", "универсальная"),
    "сумка":               (1, False, "casual", "универсальная"),
    "рюкзак":              (1, False, "casual", "универсальная"),
}


# ---------- Определение цвета ----------

def _rgb_to_color_name(r: int, g: int, b: int) -> str:
    h, s, v = colorsys.rgb_to_hsv(r / 255, g / 255, b / 255)
    h_deg = h * 360
    s_pct = s * 100
    v_pct = v * 100

    if s_pct < 10:
        if v_pct < 15:
            return "чёрный"
        if v_pct < 40:
            return "тёмно-серый"
        if v_pct < 70:
            return "серый"
        if v_pct < 90:
            return "светло-серый"
        return "белый"

    if v_pct > 88 and s_pct < 35:
        return "кремовый"

    if 15 <= h_deg < 60 and s_pct < 45 and v_pct > 45:
        return "бежевый"

    if 10 <= h_deg < 55 and v_pct < 45:
        if v_pct < 25:
            return "тёмно-коричневый"
        return "коричневый"

    if 200 <= h_deg < 260 and v_pct < 45:
        return "тёмно-синий"

    if h_deg < 15 or h_deg >= 345:
        if v_pct < 50:
            return "бордовый"
        if v_pct > 75 and s_pct < 50:
            return "розовый"
        return "красный"

    if h_deg < 40:
        if s_pct < 50 and v_pct < 65:
            return "терракотовый"
        return "оранжевый"

    if h_deg < 55:
        if v_pct < 60:
            return "горчичный"
        return "жёлтый"

    if h_deg < 165:
        if v_pct < 50:
            return "оливковый"
        if s_pct < 35:
            return "хаки"
        if v_pct > 80 and s_pct < 50:
            return "мятный"
        return "зелёный"

    if h_deg < 200:
        if s_pct > 50 and v_pct < 70:
            return "бирюзовый"
        return "голубой"

    if h_deg < 255:
        return "синий"

    if h_deg < 290:
        if s_pct < 40 and v_pct > 70:
            return "сиреневый"
        return "фиолетовый"

    if h_deg < 345:
        if s_pct < 40 and v_pct > 70:
            return "лиловый"
        return "розовый"

    return "неизвестный"


def detect_color(image_bytes: bytes) -> tuple[str, list[str]]:
    img = Image.open(BytesIO(image_bytes)).convert("RGB")
    img.thumbnail((200, 200))

    w, h = img.size
    margin_x = int(w * 0.20)
    margin_y = int(h * 0.20)
    img = img.crop((margin_x, margin_y, w - margin_x, h - margin_y))

    pixels = list(img.getdata())
    if not pixels:
        return "неизвестный", []

    dark_neutral = 0
    for p in pixels:
        r, g, b = p
        _, s, v = colorsys.rgb_to_hsv(r / 255, g / 255, b / 255)
        if v < 0.22 and s < 0.25:
            dark_neutral += 1

    dark_ratio = dark_neutral / len(pixels)
    if dark_ratio > 0.35:
        logger.info("detect_color: dark_ratio=%.2f → чёрный", dark_ratio)
        return "чёрный", ["чёрный", "тёмно-серый", "тёмно-синий"]

    colored_pixels = []
    for p in pixels:
        r, g, b = p
        _, s, v = colorsys.rgb_to_hsv(r / 255, g / 255, b / 255)
        if v < 0.25:
            is_colored = s >= 0.30
        else:
            is_colored = s >= 0.12
        if is_colored:
            colored_pixels.append(p)

    colored_ratio = len(colored_pixels) / len(pixels)
    source_pixels = colored_pixels if colored_ratio >= 0.15 else pixels

    name_weights: dict[str, float] = {}
    for p in source_pixels:
        r, g, b = p
        _, s, _ = colorsys.rgb_to_hsv(r / 255, g / 255, b / 255)
        name = _rgb_to_color_name(r, g, b)
        weight = 1 + 4 * s
        name_weights[name] = name_weights.get(name, 0.0) + weight

    if not name_weights:
        return "неизвестный", []

    sorted_names = sorted(name_weights.items(), key=lambda x: -x[1])
    primary = sorted_names[0][0]
    top3 = [name for name, _ in sorted_names[:3]]

    logger.info(
        "detect_color: colored_ratio=%.2f dark_ratio=%.2f primary=%s top3=%s",
        colored_ratio, dark_ratio, primary, top3,
    )
    return primary, top3


# ---------- Загрузка SigLIP ----------

_siglip_model = None
_siglip_processor = None
_siglip_text_embeds = None


def _get_siglip():
    global _siglip_model, _siglip_processor, _siglip_text_embeds
    if _siglip_model is None:
        logger.info("Загружаем SigLIP %s...", SIGLIP_MODEL_ID)
        _siglip_model = AutoModel.from_pretrained(SIGLIP_MODEL_ID)
        _siglip_model.eval()
        _siglip_processor = AutoProcessor.from_pretrained(SIGLIP_MODEL_ID)

        logger.info("Считаем текстовые эмбеддинги SigLIP...")
        text_inputs = _siglip_processor(
            text=ENGLISH_LABELS,
            return_tensors="pt",
            padding="max_length",
        )
        with torch.no_grad():
            text_embeds = _siglip_model.get_text_features(**text_inputs)
            text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
            _siglip_text_embeds = text_embeds

        logger.info("SigLIP загружен, эмбеддинги посчитаны")
    return _siglip_model, _siglip_processor, _siglip_text_embeds


def _classify_type_siglip(image: Image.Image, image_bytes: bytes) -> list[dict]:
    cached = _siglip_cache_get(image_bytes)
    if cached is not None:
        return cached

    model, processor, text_embeds = _get_siglip()

    image_inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        image_embeds = model.get_image_features(**image_inputs)
        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
        logits = image_embeds @ text_embeds.T * model.logit_scale.exp()
        probs = logits.softmax(dim=-1)[0]

    results = [
        {"label": label, "score": float(prob)}
        for label, prob in zip(ENGLISH_LABELS, probs.tolist())
    ]
    results.sort(key=lambda x: x["score"], reverse=True)

    options: list[dict] = []
    seen_types: set[str] = set()
    for r in results:
        en_label = r["label"]
        score = r["score"]
        if en_label in NEGATIVE_LABELS:
            continue
        entry = LABEL_MAP.get(en_label)
        if entry is None:
            continue
        ru_type, category = entry
        if ru_type in seen_types:
            continue
        seen_types.add(ru_type)
        defaults = TYPE_DEFAULTS.get(ru_type, (3, False, "casual", "универсальная"))
        warmth, waterproof, formal, season = defaults
        options.append({
            "type": ru_type,
            "category": category,
            "confidence": score,
            "warmth_level": warmth,
            "waterproof": waterproof,
            "formal_level": formal,
            "season": season,
        })
        if len(options) >= 3:
            break

    _siglip_cache_set(image_bytes, options)
    return options


# ---------- Прогрев ----------

_warmed_up = False


def warmup() -> None:
    global _warmed_up
    if _warmed_up:
        logger.info("Прогрев уже выполнен, пропускаем")
        return

    logger.info("Прогрев моделей начался...")
    _get_siglip()
    warmup_segmentation()

    dummy_img = Image.new("RGB", (384, 384), color=(128, 128, 128))
    dummy_bytes = BytesIO()
    dummy_img.save(dummy_bytes, format="JPEG")
    dummy_data = dummy_bytes.getvalue()

    _classify_type_siglip(dummy_img, dummy_data)

    _warmed_up = True
    logger.info("Прогрев моделей завершён")


# ---------- Главная функция ----------

async def classify_clothing(image_bytes: bytes) -> dict:
    """Распознать тип (SigLIP) и цвет (HSV), предварительно вырезав вещь."""
    # 1. Сегментация: обрезаем по bbox вещи
    cropped_bytes = crop_garment(image_bytes)

    # 2. Цвет по обрезанному изображению
    color_primary, color_candidates = detect_color(cropped_bytes)

    try:
        image = Image.open(BytesIO(cropped_bytes)).convert("RGB")
    except Exception:
        logger.exception("Не удалось открыть изображение")
        return {
            "ok": False,
            "color": color_primary,
            "color_candidates": color_candidates,
        }

    try:
        options = _classify_type_siglip(image, cropped_bytes)
    except Exception:
        logger.exception("Ошибка SigLIP")
        return {
            "ok": False,
            "color": color_primary,
            "color_candidates": color_candidates,
        }

    if not options or options[0]["confidence"] < SIGLIP_THRESHOLD:
        logger.info(
            "SigLIP: нет уверенных вариантов (top=%.3f < %.3f)",
            options[0]["confidence"] if options else 0.0,
            SIGLIP_THRESHOLD,
        )
        return {
            "ok": False,
            "color": color_primary,
            "color_candidates": color_candidates,
        }

    top = options[0]
    logger.info(
        "Распознавание: top=%s (%.3f) category=%s color=%s, вариантов=%d",
        top["type"], top["confidence"], top["category"],
        color_primary, len(options),
    )

    return {
        "ok": True,
        "category": top["category"],
        "type": top["type"],
        "confidence": top["confidence"],
        "color": color_primary,
        "color_candidates": color_candidates,
        "warmth_level": top["warmth_level"],
        "waterproof": top["waterproof"],
        "formal_level": top["formal_level"],
        "season": top["season"],
        "options": options,
    }