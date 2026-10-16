"""Сегментация одежды — вырезает главную вещь из фона через SegFormer."""

import logging
from collections import deque
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

MIN_GARMENT_AREA = 0.20
DILATION_RADIUS = 15


def _garment_class_ids(model) -> set[int]:
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


def _dilate_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    result = mask.copy()
    step = max(1, radius // 3)
    for dy in range(-radius, radius + 1, step):
        for dx in range(-radius, radius + 1, step):
            if dx == 0 and dy == 0:
                continue
            shifted = np.roll(np.roll(mask, dy, axis=0), dx, axis=1)
            result |= shifted
    return result


def _largest_component_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    h, w = mask.shape
    dilated = _dilate_mask(mask, DILATION_RADIUS)

    visited = np.zeros_like(dilated, dtype=bool)
    best_size = 0
    best_bbox = None

    for y0 in range(0, h, 4):
        for x0 in range(0, w, 4):
            if not dilated[y0, x0] or visited[y0, x0]:
                continue

            queue = deque([(y0, x0)])
            visited[y0, x0] = True
            size = 0
            y_min, y_max = y0, y0
            x_min, x_max = x0, x0

            while queue:
                y, x = queue.popleft()
                size += 1
                if y < y_min: y_min = y
                if y > y_max: y_max = y
                if x < x_min: x_min = x
                if x > x_max: x_max = x

                for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w:
                        if dilated[ny, nx] and not visited[ny, nx]:
                            visited[ny, nx] = True
                            queue.append((ny, nx))

            if size > best_size:
                best_size = size
                best_bbox = (y_min, x_min, y_max, x_max)

    if best_bbox is None:
        return None

    y_min, x_min, y_max, x_max = best_bbox
    sub_mask = mask[y_min:y_max + 1, x_min:x_max + 1]
    ys, xs = np.where(sub_mask)
    if len(ys) == 0:
        return best_bbox

    return (
        int(ys.min()) + y_min,
        int(xs.min()) + x_min,
        int(ys.max()) + y_min,
        int(xs.max()) + x_min,
    )


def crop_garment(image_bytes: bytes) -> tuple[bytes, np.ndarray | None]:
    """Вырезать главную вещь.

    Возвращает (cropped_image_bytes, mask):
      - cropped_image_bytes — байты JPEG обрезанного изображения;
      - mask — bool-массив (H, W) размера обрезанного изображения,
        где True — пиксель вещи. None — если сегментация не сработала.
    """
    try:
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        original_size = image.size
        w, h = original_size

        model, processor, garment_ids = _get_model()
        if not garment_ids:
            logger.info("Сегментация: классы одежды не найдены, оригинал")
            return image_bytes, None

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
            logger.info("Сегментация: вещь не найдена, оригинал")
            return image_bytes, None

        bbox = _largest_component_bbox(garment_mask)
        if bbox is None:
            logger.info("Сегментация: не удалось выделить компоненту, оригинал")
            return image_bytes, None

        y_min, x_min, y_max, x_max = bbox

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
                "Сегментация: главная вещь %.0f%% кадра, оригинал",
                area_ratio * 100,
            )
            return image_bytes, None

        cropped = image.crop((x_min, y_min, x_max, y_max))
        crop_mask = garment_mask[y_min:y_max, x_min:x_max]

        buf = BytesIO()
        cropped.save(buf, format="JPEG", quality=92)

        logger.info(
            "Сегментация: %s → %s (главная вещь %.0f%% кадра)",
            original_size, cropped.size, area_ratio * 100,
        )
        return buf.getvalue(), crop_mask

    except Exception:
        logger.exception("Ошибка сегментации, использую оригинал")
        return image_bytes, None


def warmup_segmentation() -> None:
    dummy = Image.new("RGB", (384, 384), color=(200, 200, 200))
    buf = BytesIO()
    dummy.save(buf, format="JPEG")
    crop_garment(buf.getvalue())