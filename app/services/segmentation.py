"""Сегментация одежды — вырезает главную вещь из фона через SegFormer.

Алгоритм:
  1. Модель выдаёт попиксельную маску классов.
  2. Оставляем только классы одежды (по id2label).
  3. Применяем дилатацию, чтобы склеить близкие части (рукав + торс).
  4. Разбиваем на связные компоненты, берём самую крупную.
  5. Обрезаем по её bbox с отступом.
"""

import logging
from io import BytesIO

import numpy as np
import torch
from PIL import Image
from scipy import ndimage
from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation

logger = logging.getLogger(__name__)

MODEL_ID = "sayeed99/segformer_b3_clothes"

_model = None
_processor = None
_garment_ids: set[int] | None = None

# Минимальная доля кадра, которую должна занимать главная вещь.
# Если меньше — считаем, что сегментация не справилась, и оставляем оригинал.
MIN_GARMENT_AREA = 0.20

# Радиус дилатации маски (в пикселях) — склеивает близкие части вещи.
DILATION_RADIUS = 15


def _garment_class_ids(model) -> set[int]:
    """Определить id классов, соответствующих одежде, по id2label."""
    id2label = getattr(model.config, "id2label", {}) or {}
    garment_keywords = {
        "upper", "lower", "dress", "skirt", "pant", "trouser",
        "shoe", "boot", "hat", "cap", "scarf", "belt", "bag",
        "cloth", "glove", "sock", "tie",
    }
    exclude_keywords = {
        "background", "hair", "face", "arm", "leg", "skin",
        "sunglasses", "head",
    }
    ids = set()
    for class_id, label in id2label.items():
        low = str(label).lower()
        if any(kw in low for kw in exclude_keywords):
            continue
        if any(kw in low for kw in garment_keywords):
            ids.add(int(class_id))
    return ids


def _get_model():
    global _model, _processor, _garment_ids
    if _model is None:
        logger.info("Загружаем сегментацию %s...", MODEL_ID)
        _processor = AutoImageProcessor.from_pretrained(MODEL_ID)
        _model = AutoModelForSemanticSegmentation.from_pretrained(MODEL_ID)
        _model.eval()
        _garment_ids = _garment_class_ids(_model)
        logger.info(
            "Сегментация загружена. Классы одежды: %s",
            sorted(_garment_ids),
        )
    return _model, _processor, _garment_ids


def _largest_component_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    """Найти bbox самой крупной связной компоненты в бинарной маске.

    Возвращает (y_min, x_min, y_max, x_max) или None.
    """
    # Дилатация — склеиваем близкие части (рукава + торс, юбка + топ).
    structure = np.ones((DILATION_RADIUS, DILATION_RADIUS), dtype=bool)
    dilated = ndimage.binary_dilation(mask, structure=structure)

    # Метки связных компонент.
    labeled, num_features = ndimage.label(dilated)
    if num_features == 0:
        return None

    # Площадь каждой компоненты.
    sizes = ndimage.sum(dilated, labeled, range(1, num_features + 1))
    if len(sizes) == 0:
        return None

    # Индекс самой крупной (нумерация с 1).
    largest_idx = int(np.argmax(sizes)) + 1
    largest_mask = labeled == largest_idx

    # Возвращаемся к исходной маске (без дилатации), чтобы bbox был точнее.
    # Пересекаем крупную компоненту с оригинальной маской.
    final_mask = largest_mask & mask
    if not final_mask.any():
        # Если пересечение пусто (редкий случай) — используем расширенную.
        final_mask = largest_mask

    ys, xs = np.where(final_mask)
    if len(ys) == 0:
        return None

    return int(ys.min()), int(xs.min()), int(ys.max()), int(xs.max())


def crop_garment(image_bytes: bytes) -> bytes:
    """Найти главную вещь на фото и вернуть обрезанное изображение.

    Если сегментация не справилась — возвращает оригинал.
    """
    try:
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        original_size = image.size
        w, h = original_size

        model, processor, garment_ids = _get_model()
        if not garment_ids:
            logger.info("Сегментация: классы одежды не найдены, оставляю оригинал")
            return image_bytes

        inputs = processor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
            logits = torch.nn.functional.interpolate(
                outputs.logits,
                size=(h, w),
                mode="bilinear",
                align_corners=False,
            )
            mask = logits.argmax(dim=1)[0].cpu().numpy()

        garment_mask = np.isin(mask, list(garment_ids))
        if not garment_mask.any():
            logger.info("Сегментация: вещь не найдена, оставляю оригинал")
            return image_bytes

        bbox = _largest_component_bbox(garment_mask)
        if bbox is None:
            logger.info("Сегментация: не удалось выделить компоненту, оригинал")
            return image_bytes

        y_min, x_min, y_max, x_max = bbox

        # Отступ 5% от размера bbox.
        pad_y = max(2, int((y_max - y_min) * 0.05))
        pad_x = max(2, int((x_max - x_min) * 0.05))
        y_min = max(0, y_min - pad_y)
        y_max = min(h, y_max + pad_y)
        x_min = max(0, x_min - pad_x)
        x_max = min(w, x_max + pad_x)

        crop_w = x_max - x_min
        crop_h = y_max - y_min
        area_ratio = (crop_w * crop_h) / (w * h)

        if area_ratio < MIN_GARMENT_AREA:
            logger.info(
                "Сегментация: главная вещь занимает %.0f%% кадра, оригинал",
                area_ratio * 100,
            )
            return image_bytes

        cropped = image.crop((x_min, y_min, x_max, y_max))
        buf = BytesIO()
        cropped.save(buf, format="JPEG", quality=92)

        logger.info(
            "Сегментация: %s → %s (главная вещь %.0f%% кадра)",
            original_size, cropped.size, area_ratio * 100,
        )
        return buf.getvalue()

    except Exception:
        logger.exception("Ошибка сегментации, использую оригинал")
        return image_bytes


def warmup_segmentation() -> None:
    """Прогреть сегментацию dummy-изображением."""
    dummy = Image.new("RGB", (384, 384), color=(200, 200, 200))
    buf = BytesIO()
    dummy.save(buf, format="JPEG")
    crop_garment(buf.getvalue())