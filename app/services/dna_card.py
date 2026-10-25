"""Генерация виральной ДНК-карточки в стиле модного журнала."""

import io
import logging
from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger(__name__)

# Формат 4:5 (Идеально для Instagram, Stories и Telegram)
CARD_W = 1080
CARD_H = 1350
MARGIN = 80

# Шрифты (В Dockerfile установлен fonts-dejavu-core)
FONT_SERIF_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf"
FONT_SANS_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FONT_SANS_REG = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
# Заменили курсив на обычный шрифт, так как в fonts-dejavu-core нет Oblique
FONT_SANS_ITALIC = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"

# Эстетичная палитра (Vogue-style)
BG_COLOR = (248, 246, 240)      # Теплый бумажный белый
TEXT_BLACK = (15, 15, 15)       # Насыщенный черный
TEXT_GREY = (120, 120, 120)     # Серый для подписей
ACCENT_RED = (190, 40, 45)      # Журнальный красный
WHITE = (255, 255, 255)

# Расширенный словарь цветов для парсинга
COLOR_MAP = {
    "черный": (20, 20, 20), "чёрный": (20, 20, 20), "белый": (245, 245, 245),
    "серый": (128, 128, 128), "светло-серый": (200, 200, 200), "тёмно-серый": (80, 80, 80),
    "красный": (210, 40, 40), "бордовый": (130, 20, 30), "розовый": (240, 130, 170),
    "синий": (40, 80, 180), "голубой": (130, 190, 230), "тёмно-синий": (20, 40, 100),
    "зеленый": (50, 150, 80), "зелёный": (50, 150, 80), "оливковый": (110, 120, 50),
    "хаки": (160, 160, 100), "мятный": (170, 230, 200),
    "желтый": (240, 200, 40), "жёлтый": (240, 200, 40), "горчичный": (200, 160, 40),
    "коричневый": (110, 70, 40), "бежевый": (230, 210, 180), "кремовый": (250, 240, 220),
    "фиолетовый": (130, 60, 180), "сиреневый": (190, 160, 210),
    "оранжевый": (230, 120, 40), "терракотовый": (190, 90, 60)
}

def _load_font(path, size):
    """Безопасная загрузка шрифта с фолбэком на кириллический DejaVuSans."""
    try:
        return ImageFont.truetype(path, size)
    except Exception:
        logger.warning(f"Шрифт {path} не найден, fallback на DejaVuSans.")
        try:
            # Пытаемся загрузить базовый шрифт, который точно поддерживает кириллицу
            return ImageFont.truetype(FONT_SANS_REG, size)
        except Exception:
            # Если уж совсем ничего нет, используем дефолт (но это маловероятно)
            return ImageFont.load_default()

def _get_rgb(color_name: str) -> tuple:
    """Преобразовать название цвета в RGB."""
    c = color_name.lower().strip()
    if c in COLOR_MAP:
        return COLOR_MAP[c]
    # Частичное совпадение (например, "темно-зеленый")
    for key, rgb in COLOR_MAP.items():
        if key in c:
            return rgb
    return (180, 180, 180) # Fallback на серый

def _draw_text_wrapped(draw, text, font, fill, max_width, start_y, line_spacing=1.2):
    """Рисует текст с переносом строк, если он слишком длинный."""
    words = text.split()
    lines = []
    current_line = []

    for word in words:
        current_line.append(word)
        # Проверяем ширину текущей строки
        bbox = draw.textbbox((0, 0), " ".join(current_line), font=font)
        if bbox[2] - bbox[0] > max_width:
            current_line.pop()
            if current_line:
                lines.append(" ".join(current_line))
            current_line = [word]
    if current_line:
        lines.append(" ".join(current_line))

    y = start_y
    bbox = draw.textbbox((0, 0), "A", font=font)
    line_height = (bbox[3] - bbox[1]) * line_spacing

    for line in lines:
        draw.text((MARGIN, y), line, fill=fill, font=font)
        y += line_height

    return y # Возвращаем Y для следующего блока

def build_dna_card(
    total_items: int,
    dominant_style: str,
    top_colors: list[tuple[str, int]],
    warmth_label: str,
    gaps: list[str],
) -> bytes:
    """Сгенерировать виральную ДНК-карточку."""

    canvas = Image.new("RGB", (CARD_W, CARD_H), BG_COLOR)
    draw = ImageDraw.Draw(canvas)

    # Загружаем шрифты
    f_brand = _load_font(FONT_SERIF_BOLD, 72)
    f_sub = _load_font(FONT_SANS_REG, 24)
    f_title = _load_font(FONT_SANS_BOLD, 80)
    f_large = _load_font(FONT_SERIF_BOLD, 110)
    f_text = _load_font(FONT_SANS_REG, 32)
    f_italic = _load_font(FONT_SANS_ITALIC, 28) # Теперь это обычный шрифт
    f_cta_bold = _load_font(FONT_SANS_BOLD, 36)

    # ================= ШАПКА =================
    brand = "MyWordroobBot"
    bbox = draw.textbbox((0, 0), brand, font=f_brand)
    draw.text(((CARD_W - (bbox[2]-bbox[0])) // 2, MARGIN), brand, fill=TEXT_BLACK, font=f_brand)

    subtitle = "• СТИЛЕВОЙ ПРОФИЛЬ •"
    bbox = draw.textbbox((0, 0), subtitle, font=f_sub)
    draw.text(((CARD_W - (bbox[2]-bbox[0])) // 2, MARGIN + 90), subtitle, fill=TEXT_GREY, font=f_sub)

    # Линия
    draw.line([(MARGIN, 230), (CARD_W - MARGIN, 230)], fill=TEXT_BLACK, width=4)

    # ================= СТИЛЬ И КОЛИЧЕСТВО =================
    y = 280
    draw.text((MARGIN, y), "ДОМИНИРУЮЩИЙ СТИЛЬ", fill=TEXT_GREY, font=f_sub)

    style_ru = {"casual": "CASUAL", "business": "BUSINESS", "sport": "SPORT"}.get(dominant_style.lower(), dominant_style.upper())
    draw.text((MARGIN, y + 40), style_ru, fill=TEXT_BLACK, font=f_title)

    # Количество вещей (справа)
    items_str = str(total_items)
    bbox = draw.textbbox((0, 0), items_str, font=f_large)
    draw.text((CARD_W - MARGIN - (bbox[2]-bbox[0]), y + 20), items_str, fill=ACCENT_RED, font=f_large)
    draw.text((CARD_W - MARGIN - (bbox[2]-bbox[0]), y), "ВЕЩЕЙ В БАЗЕ", fill=TEXT_GREY, font=f_sub)

    draw.line([(MARGIN, 460), (CARD_W - MARGIN, 460)], fill=TEXT_GREY, width=1)

    # ================= ЦВЕТА =================
    y = 500
    draw.text((MARGIN, y), "ЦВЕТОВАЯ ПАЛИТРА", fill=TEXT_GREY, font=f_sub)
    y += 50

    # Рисуем красивые "таблетки" (pill shapes) с цветами
    pill_w = 160
    pill_h = 80
    gap = 20
    x = MARGIN

    for color_name, _ in top_colors[:4]:
        rgb = _get_rgb(color_name)
        # Сама плашка
        draw.rounded_rectangle([(x, y), (x + pill_w, y + pill_h)], radius=40, fill=rgb, outline=TEXT_GREY, width=2)
        # Подпись под плашкой
        c_text = color_name.capitalize()
        bbox = draw.textbbox((0, 0), c_text, font=f_text)
        draw.text((x + (pill_w - (bbox[2]-bbox[0]))//2, y + pill_h + 10), c_text, fill=TEXT_BLACK, font=f_text)
        x += pill_w + gap

    y += pill_h + 60
    draw.line([(MARGIN, y), (CARD_W - MARGIN, y)], fill=TEXT_GREY, width=1)

    # ================= ТЕПЛО И ПРОБЕЛЫ =================
    y += 40
    draw.text((MARGIN, y), "ТЕМПЕРАТУРНЫЙ ПРОФИЛЬ", fill=TEXT_GREY, font=f_sub)
    y += 40
    draw.text((MARGIN, y), warmth_label.capitalize(), fill=TEXT_BLACK, font=f_text)

    y += 80
    draw.text((MARGIN, y), "ЧТО СТОИТ ДОКУПИТЬ", fill=TEXT_GREY, font=f_sub)
    y += 50

    if gaps:
        for gap_text in gaps[:3]:
            bullet_text = f"— {gap_text.capitalize()}"
            y = _draw_text_wrapped(draw, bullet_text, f_text, TEXT_BLACK, CARD_W - MARGIN*2, y)
            y += 10
    else:
        draw.text((MARGIN, y), "— Идеальный гардероб, докупать нечего!", fill=TEXT_BLACK, font=f_text)

    # ================= ФУТЕР (CTA) =================
    # Прижимаем к низу
    cta_y = CARD_H - MARGIN - 100

    # Текст-призыв
    invite = "Узнай свой стиль в Telegram:"
    bbox = draw.textbbox((0, 0), invite, font=f_text)
    draw.text(((CARD_W - (bbox[2]-bbox[0])) // 2, cta_y - 50), invite, fill=TEXT_GREY, font=f_text)

    # Кнопка с ссылкой
    link_text = "@MyWordroobBot"
    bbox = draw.textbbox((0, 0), link_text, font=f_cta_bold)
    btn_w = (bbox[2]-bbox[0]) + 100
    btn_h = 80
    btn_x = (CARD_W - btn_w) // 2

    draw.rounded_rectangle(
        [(btn_x, cta_y), (btn_x + btn_w, cta_y + btn_h)],
        radius=15, fill=TEXT_BLACK
    )

    draw.text(
        (btn_x + 50, cta_y + 20),
        link_text, fill=WHITE, font=f_cta_bold
    )

    out = io.BytesIO()
    canvas.save(out, format="JPEG", quality=95)
    return out.getvalue()
