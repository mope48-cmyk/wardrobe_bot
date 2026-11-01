"""Тесты определения цвета (app/services/vision.py: detect_color, _rgb_to_color_name)."""

from io import BytesIO

import numpy as np
from PIL import Image

from app.services.vision import _rgb_to_color_name, detect_color


def _image_from_pixels(pixels, size=(40, 40)):
    arr = np.array(pixels, dtype=np.uint8).reshape((size[1], size[0], 3))
    img = Image.fromarray(arr, "RGB")
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_dark_desaturated_is_black():
    px = [(12, 12, 12)] * (40 * 40)
    primary, _, detected = detect_color(_image_from_pixels(px))
    assert primary == "чёрный"
    assert detected == ["чёрный"]


def test_dark_saturated_blue_stays_dark_blue():
    # Тёмно-синий (насыщенный) не должен уходить в «чёрный»
    px = [(5, 10, 60)] * (40 * 40)
    primary, _, detected = detect_color(_image_from_pixels(px))
    assert primary == "тёмно-синий", primary
    assert detected == ["тёмно-синий"]


def test_dark_saturated_brown_stays_brown():
    px = [(40, 20, 8)] * (40 * 40)
    primary, _, _ = detect_color(_image_from_pixels(px))
    assert primary in ("коричневый", "тёмно-коричневый"), primary


def test_warm_light_gray_is_not_beige():
    # Тёплый светло-серый (низкая насыщенность) остаётся серым
    px = [(190, 185, 175)] * (40 * 40)
    primary, _, detected = detect_color(_image_from_pixels(px))
    assert primary in ("серый", "светло-серый"), primary
    assert "бежевый" not in detected


def test_true_beige_stays_beige():
    px = [(210, 190, 160)] * (40 * 40)
    primary, _, _detected = detect_color(_image_from_pixels(px))
    assert primary == "бежевый", primary


def test_blue_with_shadows_stays_blue():
    # «Джинсы в тени»: 55% синего + 45% почти чёрных теней.
    # Чёрный не должен побеждать — доминирует синий.
    px = ([(40, 90, 180)] * 880) + ([(8, 8, 10)] * 720)
    primary, _, detected = detect_color(_image_from_pixels(px))
    assert primary == "синий", primary
    assert "чёрный" not in detected


def test_dark_denim_stays_dark_blue():
    # Тёмные джинсы (насыщенный тёмно-синий) — не чёрный.
    px = [(18, 25, 90)] * (40 * 40)
    primary, _, _detected = detect_color(_image_from_pixels(px))
    assert primary in ("синий", "тёмно-синий"), primary


def test_rgb_to_color_name_achromatic_threshold():
    # Ниже порога s<16 — серый, а не бежевый
    assert _rgb_to_color_name(190, 185, 175) == "светло-серый"
    assert _rgb_to_color_name(120, 120, 120) == "серый"
