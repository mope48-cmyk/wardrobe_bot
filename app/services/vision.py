"""Распознавание одежды через SigLIP (тип) и Florence-2 (материал/толщина).

Цвет определяется отдельно через HSV-анализ пикселей.

Модели загружаются лениво — по первому фото. Каждая модель кэшируется
в памяти контейнера до перезапуска.
"""

import colorsys
import logging
from io import BytesIO

import torch
from PIL import Image
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoProcessor,
)

logger = logging.getLogger(__name__)

# ---------- Модели ----------

SIGLIP_MODEL_ID = "google/siglip-so400m-patch14-384"
FLORENCE_MODEL_ID = "microsoft/Florence-2-large-ft"

# Порог уверенности SigLIP: если топ-1 ниже — считаем распознавание неудачным
SIGLIP_THRESHOLD = 0.15


# ---------- Метки для SigLIP ----------

ENGLISH_LABELS = [
    # Верхняя одежда
    "a jacket", "a winter jacket",
    "a coat", "a long coat",
    "a raincoat", "a waterproof jacket",
    "a windbreaker",
    "a puffer jacket",
    "a leather jacket",
    "a denim jacket",
    "a suit jacket", "a blazer",
    "a bomber jacket", "a parka", "a trench coat",
    # Свитеры и средний слой
    "a sweater", "a knitted sweater", "a wool sweater",
    "a hoodie", "a sweatshirt", "a cardigan",
    "a vest", "a turtleneck", "a long-sleeve shirt",
    # Лёгкий верх
    "a shirt", "a t-shirt", "a polo shirt",
    "a blouse", "a top", "a tank top",
    "a dress",
    # Низ
    "trousers", "jeans", "shorts",
    "a skirt", "leggings", "sweatpants", "joggers",
    # Обувь
    "sneakers", "boots", "shoes", "sandals", "heels",
    # Аксессуары
    "a hat", "a beanie", "a cap", "a scarf", "gloves",
    "a belt", "a bag", "a backpack",
    # НЕГАТИВНЫЕ
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
    "boots": ("ботинки", "обувь"),
    "shoes": ("туфли", "обувь"),
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


# ---------- TYPE_DEFAULTS ----------

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
    "ботинки":             (3, True,  "casual", "демисезон"),
    "туфли":               (2, False, "business", "универсальная"),
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


# ---------- Материалы (английский → русский) ----------

MATERIAL_MAP = {
    "cotton": "хлопок",
    "wool": "шерсть",
    "leather": "кожа",
    "faux leather": "экокожа",
    "denim": "деним",
    "silk": "шёлк",
    "polyester": "полиэстер",
    "linen": "лён",
    "knit": "трикотаж",
    "knitted": "трикотаж",
    "corduroy": "вельвет",
    "suede": "замша",
    "cashmere": "кашемир",
    "viscose": "вискоза",
    "nylon": "нейлон",
    "fleece": "флис",
    "velvet": "бархат",
    "fur": "мех",
    "satin": "атлас",
    "chiffon": "шифон",
}

THICKNESS_MAP = {
    "thin": "тонкая",
    "lightweight": "тонкая",
    "light": "тонкая",
    "medium": "средняя",
    "thick": "толстая",
    "heavy": "толстая",
}


# ---------- Определение цвета ----------

def _rgb_to_color_name(r: int, g: int, b: int) -> str:
    """Классификация цвета по HSV."""
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
    """Определить доминирующий цвет и топ-3."""
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


# ---------- Загрузка моделей (ленивая) ----------

_siglip_model = None
_siglip_processor = None
_florence_model = None
_florence_processor = None


def _get_siglip():
    """Загрузить SigLIP один раз. Возвращает (model, processor)."""
    global _siglip_model, _siglip_processor
    if _siglip_model is None:
        logger.info("Загружаем SigLIP %s...", SIGLIP_MODEL_ID)
        _siglip_model = AutoModel.from_pretrained(SIGLIP_MODEL_ID)
        _siglip_model.eval()
        _siglip_processor = AutoProcessor.from_pretrained(SIGLIP_MODEL_ID)
        logger.info("SigLIP загружен")
    return _siglip_model, _siglip_processor


def _get_florence():
    """Загрузить Florence-2 один раз. Возвращает (model, processor)."""
    global _florence_model, _florence_processor
    if _florence_model is None:
        logger.info("Загружаем Florence-2 %s...", FLORENCE_MODEL_ID)
        _florence_model = AutoModelForCausalLM.from_pretrained(
            FLORENCE_MODEL_ID, trust_remote_code=True
        )
        _florence_model.eval()
        _florence_processor = AutoProcessor.from_pretrained(
            FLORENCE_MODEL_ID, trust_remote_code=True
        )
        logger.info("Florence-2 загружен")
    return _florence_model, _florence_processor


# ---------- Классификация типа через SigLIP ----------

def _classify_type_siglip(image: Image.Image) -> list[dict]:
    """Вернуть топ-3 типа с уверенностью."""
    model, processor = _get_siglip()

    inputs = processor(
        text=ENGLISH_LABELS,
        images=image,
        return_tensors="pt",
        padding="max_length",
    )

    with torch.no_grad():
        outputs = model(**inputs)
        probs = outputs.logits_per_image.softmax(dim=1)[0]

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

    return options


# ---------- Извлечение атрибутов через Florence ----------

def _ask_florence(model, processor, image: Image.Image, prompt: str) -> str:
    """Задать один VQA-вопрос Florence-2, вернуть ответ строкой."""
    inputs = processor(text=prompt, images=image, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=30,
            num_beams=3,
        )
    text = processor.batch_decode(out, skip_special_tokens=False)[0]
    parsed = processor.post_process_generation(
        text, task=prompt, image_size=(image.width, image.height)
    )
    return str(list(parsed.values())[0]).strip().lower()


def _extract_attributes_florence(image: Image.Image) -> dict:
    """Извлечь материал и толщину через Florence-2.

    Возвращает:
    {
        "material_en": "cotton" | "",
        "material_ru": "хлопок" | "",
        "thickness": "тонкая" | "средняя" | "толстая" | "",
    }
    """
    model, processor = _get_florence()

    try:
        material_raw = _ask_florence(
            model, processor, image,
            "<VQA>What material is this garment made of?"
        )
        thickness_raw = _ask_florence(
            model, processor, image,
            "<VQA>Is this garment thick or thin?"
        )
    except Exception:
        logger.exception("Ошибка Florence-2")
        return {"material_en": "", "material_ru": "", "thickness": ""}

    # Парсим материал: ищем известное слово в ответе
    material_en = ""
    for key in MATERIAL_MAP:
        if key in material_raw:
            material_en = key
            break

    thickness = ""
    for key, ru in THICKNESS_MAP.items():
        if key in thickness_raw:
            thickness = ru
            break

    logger.info(
        "Florence: material_raw=%r → %s, thickness_raw=%r → %s",
        material_raw, material_en, thickness_raw, thickness,
    )

    return {
        "material_en": material_en,
        "material_ru": MATERIAL_MAP.get(material_en, ""),
        "thickness": thickness,
    }


# ---------- Главная функция ----------

async def classify_clothing(image_bytes: bytes) -> dict:
    """Распознать вещь: тип (SigLIP), материал и толщина (Florence), цвет (HSV)."""
    # 1. Цвет (без моделей)
    color_primary, color_candidates = detect_color(image_bytes)

    try:
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
    except Exception:
        logger.exception("Не удалось открыть изображение")
        return {
            "ok": False,
            "color": color_primary,
            "color_candidates": color_candidates,
        }

    # 2. Тип через SigLIP
    try:
        options = _classify_type_siglip(image)
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

    # 3. Атрибуты через Florence
    attrs = _extract_attributes_florence(image)

    top = options[0]
    logger.info(
        "Распознавание: top=%s (%.3f) category=%s color=%s "
        "material=%s thickness=%s, вариантов=%d",
        top["type"], top["confidence"], top["category"],
        color_primary, attrs["material_ru"], attrs["thickness"],
        len(options),
    )

    return {
        "ok": True,
        "category": top["category"],
        "type": top["type"],
        "confidence": top["confidence"],
        "color": color_primary,
        "color_candidates": color_candidates,
        "material": attrs["material_ru"],
        "thickness": attrs["thickness"],
        "warmth_level": top["warmth_level"],
        "waterproof": top["waterproof"],
        "formal_level": top["formal_level"],
        "season": top["season"],
        "options": options,
    }