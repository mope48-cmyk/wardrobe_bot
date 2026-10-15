"""Сегментация одежды — вырезает вещь из фона через SegFormer."""

import logging
from io import BytesIO

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation

logger = logging.getLogger(__name__)

MODEL_ID = "sayeed99/segformer_b3_clothes"

_model = None
_processor = None
_garment_ids: set[int] | None = None


def _garment_class_ids(model) -> set[int]:
    """Определить id классов, соответствующих одежде, по id2label.

    Так код не зависит от конкретных id в модели — если модель обновят
    и переназначат классы, мы всё равно найдём нужные.
    """
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


def crop_garment(image_bytes: bytes) -> bytes:
    """Найти вещь на фото и вернуть обрезанное изображение по bbox.

    Если сегментация находит слишком маленькую область (<30% площади) —
    возвращает исходное изображение. Если модель падает — тоже исходное.
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
        ys, xs = np.where(garment_mask)
        if len(ys) == 0:
            logger.info("Сегментация: вещь не найдена, оставляю оригинал")
            return image_bytes

        y_min, y_max = int(ys.min()), int(ys.max())
        x_min, x_max = int(xs.min()), int(xs.max())

        # Отступ 5% по каждой стороне
        pad_y = max(2, int((y_max - y_min) * 0.05))
        pad_x = max(2, int((x_max - x_min) * 0.05))
        y_min = max(0, y_min - pad_y)
        y_max = min(h, y_max + pad_y)
        x_min = max(0, x_min - pad_x)
        x_max = min(w, x_max + pad_x)

        crop_w = x_max - x_min
        crop_h = y_max - y_min
        if crop_w * crop_h < 0.30 * w * h:
            logger.info(
                "Сегментация: обрезка слишком агрессивная (%sx%s), оставляю оригинал",
                crop_w, crop_h,
            )
            return image_bytes

        cropped = image.crop((x_min, y_min, x_max, y_max))
        buf = BytesIO()
        cropped.save(buf, format="JPEG", quality=92)

        logger.info("Сегментация: %s → %s", original_size, cropped.size)
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
