"""Распознавание одежды и определение цвета."""

import colorsys
import logging
from io import BytesIO

import open_clip
import torch
from PIL import Image

logger = logging.getLogger(__name__)

MODEL_NAME = "MobileCLIP2-S0"
PRETRAINED = "dfndr2b"

# Расширенный список английских меток.
ENGLISH_LABELS = [
    # Верхняя одежда
    "a jacket", "a casual jacket", "a winter jacket",
    "a coat", "a long coat", "an overcoat",
    "a raincoat", "a waterproof jacket",
    "a windbreaker",
    "a puffer jacket", "a down jacket",
    "a leather jacket",
    "a denim jacket",
    "a suit jacket", "a blazer",
    # Свитеры и лёгкий верх
    "a sweater", "a knitted sweater", "a wool sweater",
    "a hoodie", "a hooded sweatshirt",
    "a sweatshirt",
    "a cardigan",
    "a vest", "a sleeveless vest",
    # Рубашки и лёгкие
    "a shirt", "a button-up shirt", "a dress shirt",
    "a t-shirt", "a plain t-shirt",
    "a polo shirt",
    "a blouse",
    "a top", "a tank top",
    "a dress", "a summer dress", "a long dress",
    # Низ
    "trousers", "dress pants", "casual trousers",
    "jeans", "blue jeans",
    "shorts",
    "a skirt",
    "leggings",
    "sweatpants",
    # Обувь
    "sneakers", "running shoes",
    "boots", "winter boots", "ankle boots",
    "shoes", "dress shoes", "loafers",
    "sandals",
    "heels", "high heels",
    # Аксессуары
    "a hat", "a winter hat", "a beanie",
    "a cap", "a baseball cap",
    "a scarf",
    "gloves",
    "a belt",
    "a bag", "a handbag",
    "a backpack",
    # НЕГАТИВНЫЕ (не одежда)
    "a landscape", "a nature scene",
    "a room interior", "furniture",
    "food", "a meal",
    "an animal", "a pet",
    "a person's face", "a portrait",
    "a text document", "a screenshot",
    "an empty background",
]

# Английская метка → (русский тип, категория)
LABEL_MAP = {
    # Верхняя одежда
    "a jacket": ("куртка", "верх"),
    "a casual jacket": ("куртка", "верх"),
    "a winter jacket": ("зимняя куртка", "верх"),
    "a coat": ("пальто", "верх"),
    "a long coat": ("длинное пальто", "верх"),
    "an overcoat": ("пальто", "верх"),
    "a raincoat": ("дождевик", "верх"),
    "a waterproof jacket": ("непромокаемая куртка", "верх"),
    "a windbreaker": ("ветровка", "верх"),
    "a puffer jacket": ("пуховик", "верх"),
    "a down jacket": ("пуховик", "верх"),
    "a leather jacket": ("кожаная куртка", "верх"),
    "a denim jacket": ("джинсовая куртка", "верх"),
    "a suit jacket": ("пиджак", "верх"),
    "a blazer": ("блейзер", "верх"),
    # Свитеры и лёгкий верх
    "a sweater": ("свитер", "верх"),
    "a knitted sweater": ("вязаный свитер", "верх"),
    "a wool sweater": ("шерстяной свитер", "верх"),
    "a hoodie": ("толстовка", "верх"),
    "a hooded sweatshirt": ("толстовка", "верх"),
    "a sweatshirt": ("свитшот", "верх"),
    "a cardigan": ("кардиган", "верх"),
    "a vest": ("жилет", "верх"),
    "a sleeveless vest": ("безрукавка", "верх"),
    # Рубашки и лёгкие
    "a shirt": ("рубашка", "верх"),
    "a button-up shirt": ("рубашка", "верх"),
    "a dress shirt": ("классическая рубашка", "верх"),
    "a t-shirt": ("футболка", "верх"),
    "a plain t-shirt": ("футболка", "верх"),
    "a polo shirt": ("поло", "верх"),
    "a blouse": ("блузка", "верх"),
    "a top": ("топ", "верх"),
    "a tank top": ("майка", "верх"),
    "a dress": ("платье", "верх"),
    "a summer dress": ("летнее платье", "верх"),
    "a long dress": ("длинное платье", "верх"),
    # Низ
    "trousers": ("брюки", "низ"),
    "dress pants": ("классические брюки", "низ"),
    "casual trousers": ("повседневные брюки", "низ"),
    "jeans": ("джинсы", "низ"),
    "blue jeans": ("джинсы", "низ"),
    "shorts": ("шорты", "низ"),
    "a skirt": ("юбка", "низ"),
    "leggings": ("леггинсы", "низ"),
    "sweatpants": ("спортивные штаны", "низ"),
    # Обувь
    "sneakers": ("кроссовки", "обувь"),
    "running shoes": ("беговые кроссовки", "обувь"),
    "boots": ("ботинки", "обувь"),
    "winter boots": ("зимние ботинки", "обувь"),
    "ankle boots": ("полуботинки", "обувь"),
    "shoes": ("туфли", "обувь"),
    "dress shoes": ("классические туфли", "обувь"),
    "loafers": ("лоферы", "обувь"),
    "sandals": ("сандалии", "обувь"),
    "heels": ("каблуки", "обувь"),
    "high heels": ("высокие каблуки", "обувь"),
    # Аксессуары
    "a hat": ("шляпа", "аксессуар"),
    "a winter hat": ("зимняя шапка", "аксессуар"),
    "a beanie": ("шапка-бини", "аксессуар"),
    "a cap": ("кепка", "аксессуар"),
    "a baseball cap": ("бейсболка", "аксессуар"),
    "a scarf": ("шарф", "аксессуар"),
    "gloves": ("перчатки", "аксессуар"),
    "a belt": ("ремень", "аксессуар"),
    "a bag": ("сумка", "аксессуар"),
    "a handbag": ("сумка", "аксессуар"),
    "a backpack": ("рюкзак", "аксессуар"),
}

# Метки, которые означают "на фото не одежда".
NEGATIVE_LABELS = {
    "a landscape", "a nature scene",
    "a room interior", "furniture",
    "food", "a meal",
    "an animal", "a pet",
    "a person's face", "a portrait",
    "a text document", "a screenshot",
    "an empty background",
}

# Если топ-1 результат ниже этого порога — считаем распознавание неудачным.
CONFIDENCE_THRESHOLD = 0.10

# Дефолтные характеристики по русскому типу вещи.
# Формат: тип → (warmth_level, waterproof, formal_level, season)
TYPE_DEFAULTS = {
    # Верхняя одежда
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
    # Свитеры и лёгкий верх
    "свитер":              (4, False, "casual", "зима"),
    "вязаный свитер":      (4, False, "casual", "зима"),
    "шерстяной свитер":    (5, False, "casual", "зима"),
    "толстовка":           (3, False, "casual", "демисезон"),
    "свитшот":             (3, False, "casual", "демисезон"),
    "кардиган":            (3, False, "casual", "демисезон"),
    "жилет":               (2, False, "casual", "демисезон"),
    "безрукавка":          (2, False, "casual", "демисезон"),
    # Рубашки и лёгкие
    "рубашка":             (2, False, "business", "универсальная"),
    "классическая рубашка": (2, False, "business", "универсальная"),
    "футболка":            (1, False, "casual", "лето"),
    "поло":                (2, False, "casual", "лето"),
    "блузка":              (2, False, "business", "универсальная"),
    "топ":                 (1, False, "casual", "лето"),
    "майка":               (1, False, "casual", "лето"),
    "платье":              (2, False, "business", "лето"),
    "летнее платье":       (1, False, "casual", "лето"),
    "длинное платье":      (2, False, "business", "универсальная"),
    # Низ
    "брюки":               (3, False, "business", "универсальная"),
    "классические брюки":  (3, False, "business", "универсальная"),
    "повседневные брюки":  (3, False, "casual", "универсальная"),
    "джинсы":              (3, False, "casual", "универсальная"),
    "шорты":               (1, False, "casual", "лето"),
    "юбка":                (2, False, "casual", "лето"),
    "леггинсы":            (2, False, "sport", "демисезон"),
    "спортивные штаны":    (2, False, "sport", "демисезон"),
    # Обувь
    "кроссовки":           (2, False, "sport", "демисезон"),
    "беговые кроссовки":   (2, False, "sport", "лето"),
    "ботинки":             (3, True,  "casual", "демисезон"),
    "зимние ботинки":      (4, True,  "casual", "зима"),
    "полуботинки":         (3, False, "casual", "демисезон"),
    "туфли":               (2, False, "business", "универсальная"),
    "классические туфли":  (2, False, "business", "универсальная"),
    "лоферы":              (2, False, "business", "универсальная"),
    "сандалии":            (1, False, "casual", "лето"),
    "каблуки":             (2, False, "business", "универсальная"),
    "высокие каблуки":     (2, False, "business", "универсальная"),
    # Аксессуары
    "шляпа":               (2, False, "casual", "лето"),
    "зимняя шапка":        (4, False, "casual", "зима"),
    "шапка-бини":          (4, False, "casual", "зима"),
    "кепка":               (1, False, "casual", "лето"),
    "бейсболка":           (1, False, "casual", "лето"),
    "шарф":                (4, False, "casual", "зима"),
    "перчатки":            (3, False, "casual", "зима"),
    "ремень":              (1, False, "business", "универсальная"),
    "сумка":               (1, False, "casual", "универсальная"),
    "рюкзак":              (1, False, "casual", "универсальная"),
}


# ---------- Определение цвета ----------

def _auto_white_balance(img: Image.Image) -> Image.Image:
    """Gray-world: выравниваем средние значения каналов.

    Снимает цветовой сдвиг от освещения — лампа накаливания (тёплый свет)
    и уличный день (холодный) приводят к искажению, из-за которого
    алгоритм видит серый там, где на самом деле цвет.
    """
    pixels = list(img.getdata())
    if not pixels:
        return img

    n = len(pixels)
    r_avg = sum(p[0] for p in pixels) / n
    g_avg = sum(p[1] for p in pixels) / n
    b_avg = sum(p[2] for p in pixels) / n
    gray = (r_avg + g_avg + b_avg) / 3

    r_scale = gray / max(r_avg, 1)
    g_scale = gray / max(g_avg, 1)
    b_scale = gray / max(b_avg, 1)

    out = Image.new("RGB", img.size)
    out.putdata([
        (
            min(255, int(p[0] * r_scale)),
            min(255, int(p[1] * g_scale)),
            min(255, int(p[2] * b_scale)),
        )
        for p in pixels
    ])
    return out


def _rgb_to_color_name(r: int, g: int, b: int) -> str:
    """Развёрнутая классификация цвета по HSV.

    Принципы:
      - порог «серых» снижен с 15% до 10% (многие цвета умеренно насыщены);
      - добавлены промежуточные оттенки (тёмно-серый, светло-серый, navy,
        оливковый, горчичный, хаки, терракотовый, кремовый и т.д.);
      - коричневый и бежевый выделяются из оранжевого/жёлтого
        по яркости и насыщенности.
    """
    h, s, v = colorsys.rgb_to_hsv(r / 255, g / 255, b / 255)
    h_deg = h * 360
    s_pct = s * 100
    v_pct = v * 100

    # 1. Ахроматические — низкая насыщенность
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

    # 2. Коричневый — тёмный + тёплый hue (в HSV тёмно-оранжевый
    # воспринимается как коричневый, поэтому проверяем раньше оранжевого)
    if 10 <= h_deg < 55 and v_pct < 55:
        if v_pct < 30:
            return "тёмно-коричневый"
        return "коричневый"

    # 3. Бежевый — светло + слабо насыщенно + тёплый
    if 20 <= h_deg < 55 and s_pct < 40 and v_pct > 65:
        return "бежевый"

    # 4. Кремовый — очень светло + слабо насыщенно
    if v_pct > 88 and s_pct < 35:
        return "кремовый"

    # 5. Тёмно-синий / navy — тёмный + холодный синий диапазон
    if 200 <= h_deg < 260 and v_pct < 45:
        return "тёмно-синий"

    # 6. Основные цвета по hue

    # Красный / бордовый / розовый
    if h_deg < 15 or h_deg >= 345:
        if v_pct < 50:
            return "бордовый"
        if v_pct > 75 and s_pct < 50:
            return "розовый"
        return "красный"

    # Оранжевый / терракотовый
    if h_deg < 40:
        if s_pct < 50 and v_pct < 65:
            return "терракотовый"
        return "оранжевый"

    # Жёлтый / горчичный
    if h_deg < 65:
        if v_pct < 60:
            return "горчичный"
        return "жёлтый"

    # Зелёный / оливковый / хаки / мятный
    if h_deg < 165:
        if v_pct < 50:
            return "оливковый"
        if s_pct < 35:
            return "хаки"
        if v_pct > 80 and s_pct < 50:
            return "мятный"
        return "зелёный"

    # Голубой / бирюзовый
    if h_deg < 200:
        if s_pct > 50 and v_pct < 70:
            return "бирюзовый"
        return "голубой"

    # Синий
    if h_deg < 255:
        return "синий"

    # Фиолетовый / сиреневый
    if h_deg < 290:
        if s_pct < 40 and v_pct > 70:
            return "сиреневый"
        return "фиолетовый"

    # Розовый / лиловый
    if h_deg < 345:
        if s_pct < 40 and v_pct > 70:
            return "лиловый"
        return "розовый"

    return "неизвестный"


def detect_color(image_bytes: bytes) -> tuple[str, list[str]]:
    """Определить доминирующий цвет и топ-3.

    Улучшения по сравнению с базовой версией:
      1. Кадрируем 20% по краям (сильнее отсекаем фон).
      2. Корректируем баланс белого (gray-world).
      3. Квантуем в 8 доминирующих цветов вместо 5.
      4. Взвешенное голосование: цветные пиксели весят больше серых —
         это спасает, когда фон доминирует по площади, но вещь цветная.
    """
    img = Image.open(BytesIO(image_bytes)).convert("RGB")
    img.thumbnail((200, 200))

    # 1. Кадрирование 20%
    w, h = img.size
    margin_x = int(w * 0.20)
    margin_y = int(h * 0.20)
    img = img.crop((margin_x, margin_y, w - margin_x, h - margin_y))

    # 2. Баланс белого
    img = _auto_white_balance(img)

    # 3. Квантование в 8 цветов
    quantized = img.quantize(colors=8, method=Image.Quantize.MEDIANCUT)
    palette = quantized.getpalette() or []
    counts = quantized.getcolors(maxcolors=256) or []

    if not counts:
        return "неизвестный", []

    # 4. Взвешенное голосование
    name_weights: dict[str, float] = {}
    for count, idx in counts:
        r = palette[idx * 3]
        g = palette[idx * 3 + 1]
        b = palette[idx * 3 + 2]

        # Бонус за насыщенность: цветные пиксели весят до 5× больше серых.
        # Пиксель с s=0 весит 1×, с s=1 — 5×.
        _, s, _ = colorsys.rgb_to_hsv(r / 255, g / 255, b / 255)
        weight = count * (1 + 4 * s)

        name = _rgb_to_color_name(r, g, b)
        name_weights[name] = name_weights.get(name, 0.0) + weight

    sorted_names = sorted(name_weights.items(), key=lambda x: -x[1])

    primary = sorted_names[0][0] if sorted_names else "неизвестный"
    top3 = [name for name, _ in sorted_names[:3]]

    return primary, top3


# ---------- Распознавание одежды ----------

_model = None
_preprocess = None
_tokenizer = None


def _get_model():
    """Ленивая загрузка модели при первом вызове."""
    global _model, _preprocess, _tokenizer
    if _model is None:
        logger.info("Загружаем модель %s...", MODEL_NAME)
        _model, _, _preprocess = open_clip.create_model_and_transforms(
            MODEL_NAME, pretrained=PRETRAINED
        )
        _model.eval()
        _tokenizer = open_clip.get_tokenizer(MODEL_NAME)
        logger.info("Модель загружена")
    return _model, _preprocess, _tokenizer


def _build_option(en_label: str, confidence: float) -> dict | None:
    """Собрать описание одного варианта для отображения пользователю.

    Возвращает None, если метки нет в LABEL_MAP (например, это негативная
    метка или незнакомое слово).
    """
    ru_type, category = LABEL_MAP.get(en_label, (None, None))
    if ru_type is None:
        return None

    defaults = TYPE_DEFAULTS.get(ru_type, (3, False, "casual", "универсальная"))
    warmth, waterproof, formal, season = defaults

    return {
        "type": ru_type,
        "category": category,
        "confidence": confidence,
        "warmth_level": warmth,
        "waterproof": waterproof,
        "formal_level": formal,
        "season": season,
    }


async def classify_clothing(image_bytes: bytes) -> dict:
    """Распознать вещь и вернуть топ-1 и топ-3 варианта.

    Возвращает:
    {
        "ok": bool,
        "category": ...,       # топ-1 (для совместимости)
        "type": ...,           # топ-1
        "confidence": ...,     # топ-1
        "color": ...,          # определено отдельно
        "color_candidates": [...],
        "warmth_level": ...,   # дефолты для топ-1
        "waterproof": ...,
        "formal_level": ...,
        "season": ...,
        "options": [           # до 3 вариантов
            {"type": ..., "category": ..., "confidence": ...,
             "warmth_level": ..., "waterproof": ...,
             "formal_level": ..., "season": ...},
            ...,
        ],
    }
    """
    color_primary, color_candidates = detect_color(image_bytes)

    try:
        model, preprocess, tokenizer = _get_model()

        # Подготовка изображения
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        image_input = preprocess(image).unsqueeze(0)

        # Токенизация меток
        text_inputs = tokenizer(ENGLISH_LABELS)

        # Инференс
        with torch.no_grad():
            image_features = model.encode_image(image_input)
            text_features = model.encode_text(text_inputs)

            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)

        # Получаем вероятности
        probs = similarity[0].tolist()
        results = [
            {"label": label, "score": prob}
            for label, prob in zip(ENGLISH_LABELS, probs)
        ]
        results.sort(key=lambda x: x["score"], reverse=True)

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

    # Собираем топ-3 валидных варианта (пропускаем негативные метки
    # и метки без маппинга). Результаты уже отсортированы по убыванию.
    options: list[dict] = []
    seen_types: set[str] = set()
    top_confidence = 0.0

    for r in results:
        en_label = r["label"]
        score = float(r["score"])

        if en_label in NEGATIVE_LABELS:
            continue

        opt = _build_option(en_label, score)
        if opt is None:
            continue

        # Дедупликация: разные английские метки могут вести
        # на один русский тип (например, "a t-shirt" и "a plain t-shirt").
        if opt["type"] in seen_types:
            continue

        seen_types.add(opt["type"])
        options.append(opt)
        if len(options) == 1:
            top_confidence = score

        if len(options) >= 3:
            break

    if not options:
        logger.info("Нет валидных вариантов (всё негативные или низкая уверенность)")
        return {
            "ok": False,
            "color": color_primary,
            "color_candidates": color_candidates,
        }

    if top_confidence < CONFIDENCE_THRESHOLD:
        logger.info(
            "Топ-1 ниже порога: %.3f < %.3f",
            top_confidence, CONFIDENCE_THRESHOLD,
        )
        return {
            "ok": False,
            "color": color_primary,
            "color_candidates": color_candidates,
        }

    top = options[0]

    logger.info(
        "Распознавание: top=%s (%.3f) category=%s color=%s, всего вариантов=%d",
        top["type"], top["confidence"], top["category"], color_primary, len(options),
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