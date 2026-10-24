"""Генерация Vogue-коллажа из вещей образа через Pillow.

Сетка подстраивается под количество вещей:
    1 → 1×1, 2 → 2×1, 3 → 3×1, 4 → 2×2, 5–6 → 3×2, 7–9 → 3×3, 10–12 → 4×3.

Неполная последняя строка центрируется. Под каждым фото — подпись (тип).
"""

import io
import logging

from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger(__name__)

COLLAGE_W = 1080
COLLAGE_H = 1350
MARGIN = 40
GAP = 20

FONT_BRAND_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf"
FONT_SUB_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_LABEL_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"

BRAND_DEFAULT = "MyWordroobBot"

BG = (255, 255, 255)
TEXT_DARK = (17, 17, 17)
TEXT_GREY = (102, 102, 102)
TEXT_LABEL = (51, 51, 51)


def _grid_layout(n: int) -> tuple[int, int]:
    """(cols, rows) по количеству вещей."""
    if n <= 1:
        return 1, 1
    if n == 2:
        return 2, 1
    if n == 3:
        return 3, 1
    if n == 4:
        return 2, 2
    if n <= 6:
        return 3, 2
    if n <= 9:
        return 3, 3
    if n <= 12:
        return 4, 3
    return 4, (n + 3) // 4


def _try_load_font(path: str, size: int):
    try:
        return ImageFont.truetype(path, size)
    except (OSError, IOError):
        logger.warning("Шрифт %s не найден, fallback на default", path)
        try:
            return ImageFont.load_default(size=size)  # Pillow 11+
        except TypeError:
            return ImageFont.load_default()           # Pillow 10


def _paste_rgba_as_rgb(canvas: Image.Image, img: Image.Image, xy: tuple[int, int]) -> None:
    """Вставить изображение, корректно обработав прозрачность (PNG/LA → на белом)."""
    if img.mode in ("RGBA", "LA"):
        bg = Image.new("RGB", img.size, "white")
        bg.paste(img, mask=img.split()[-1])
        canvas.paste(bg, xy)
    else:
        canvas.paste(img.convert("RGB"), xy)


def build_collage(
    photos: list[bytes],
    labels: list[str],
    city: str,
    date_str: str,
    brand: str = BRAND_DEFAULT,
) -> bytes:
    """Собрать Vogue-коллаж → JPEG-байты.

    photos — байты фото вещей (по порядку)
    labels — подписи (тип вещи)
    city — название города ("" → в шапке только дата)
    date_str — дата, например "25.10.2026"
    """
    if not photos:
        raise ValueError("photos пуст — нечего рисовать")

    canvas = Image.new("RGB", (COLLAGE_W, COLLAGE_H), BG)
    draw = ImageDraw.Draw(canvas)

    font_brand = _try_load_font(FONT_BRAND_PATH, 72)
    font_sub = _try_load_font(FONT_SUB_PATH, 32)
    font_label = _try_load_font(FONT_LABEL_PATH, 28)

    # --- Шапка ---
    brand_bbox = draw.textbbox((0, 0), brand, font=font_brand)
    brand_w = brand_bbox[2] - brand_bbox[0]
    brand_h = brand_bbox[3] - brand_bbox[1]
    draw.text(
        ((COLLAGE_W - brand_w) // 2, MARGIN - brand_bbox[1]),
        brand, fill=TEXT_DARK, font=font_brand,
    )

    sub_y = MARGIN + brand_h + 20
    sub = f"{city} • {date_str}" if city else date_str
    sub_bbox = draw.textbbox((0, 0), sub, font=font_sub)
    sub_w = sub_bbox[2] - sub_bbox[0]
    sub_h = sub_bbox[3] - sub_bbox[1]
    draw.text(
        ((COLLAGE_W - sub_w) // 2, sub_y - sub_bbox[1]),
        sub, fill=TEXT_GREY, font=font_sub,
    )

    # --- Сетка ---
    grid_top = sub_y + sub_h + 40
    grid_bottom = COLLAGE_H - MARGIN
    grid_left = MARGIN
    grid_right = COLLAGE_W - MARGIN
    grid_w = grid_right - grid_left
    grid_h = grid_bottom - grid_top

    n = len(photos)
    cols, rows = _grid_layout(n)

    cell_w = (grid_w - (cols - 1) * GAP) // cols
    cell_h = (grid_h - (rows - 1) * GAP) // rows
    label_h = 40  # полоса подписи под фото

    for idx in range(n):
        photo_bytes = photos[idx]
        label = labels[idx] if idx < len(labels) else ""

        r = idx // cols
        c = idx % cols

        # Неполная последняя строка — центрируем
        cells_in_row = min(cols, n - r * cols)
        row_offset = (cols - cells_in_row) * (cell_w + GAP) // 2

        x = grid_left + row_offset + c * (cell_w + GAP)
        y = grid_top + r * (cell_h + GAP)

        try:
            img = Image.open(io.BytesIO(photo_bytes))
            img.thumbnail((cell_w, cell_h - label_h), Image.LANCZOS)
        except Exception:
            logger.exception("Битое фото, пропускаем idx=%d", idx)
            continue

        img_x = x + (cell_w - img.width) // 2
        img_y = y + (cell_h - label_h - img.height) // 2
        _paste_rgba_as_rgb(canvas, img, (img_x, img_y))

        if label:
            label_bbox = draw.textbbox((0, 0), label, font=font_label)
            label_w = label_bbox[2] - label_bbox[0]
            label_x = x + (cell_w - label_w) // 2
            label_y = y + cell_h - label_h + 8 - label_bbox[1]
            draw.text(
                (label_x, label_y), label, fill=TEXT_LABEL, font=font_label,
            )

    out = io.BytesIO()
    canvas.save(out, format="JPEG", quality=92)
    return out.getvalue()
