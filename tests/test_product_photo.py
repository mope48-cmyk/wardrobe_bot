"""Тесты товарного фото (app/services/product_photo.py)."""

from io import BytesIO

from PIL import Image

from app.services.product_photo import _catalog_with_semitransparent_bg


def _make_test_image():
    img = Image.new("RGB", (100, 100), (255, 0, 0))  # фон красный
    for y in range(40, 61):
        for x in range(40, 61):
            img.putpixel((x, y), (0, 255, 0))  # «вещь» зелёная
    mask = Image.new("L", (100, 100), 0)
    for y in range(40, 61):
        for x in range(40, 61):
            mask.putpixel((x, y), 255)
    return img, mask


def test_semitransparent_bg_keeps_item_opaque():
    img, mask = _make_test_image()
    out = _catalog_with_semitransparent_bg(img, mask, alpha=0.65)
    result = Image.open(BytesIO(out)).convert("RGB")

    # Вещь осталась исходной (непрозрачной): зелёная
    r, g, b = result.getpixel((50, 50))
    assert r < 40 and g > 220 and b < 40, (r, g, b)

    # Фон смешан с белым: красный осветлён (~255,166,166), не исходный чистый
    bg = result.getpixel((5, 5))
    assert bg[0] > 230
    assert abs(bg[1] - 166) < 30, bg
    assert abs(bg[2] - 166) < 30, bg


def test_semitransparent_bg_no_item_area_fades_fully_white():
    # Маска пустая → весь кадр смешивается с белым (композит возвращает faded)
    img = Image.new("RGB", (50, 50), (255, 0, 0))
    mask = Image.new("L", (50, 50), 0)
    out = _catalog_with_semitransparent_bg(img, mask, alpha=0.65)
    result = Image.open(BytesIO(out)).convert("RGB")
    bg = result.getpixel((25, 25))
    assert bg[0] > 230 and abs(bg[1] - 166) < 30
