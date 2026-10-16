"""Генерация товарного фото: SegFormer + rembg на белом фоне.

Гибридный подход:
  1. SegFormer находит маску конкретной вещи по категории
     (только куртку, а не шорты и ноги).
  2. rembg (u2net) даёт аккуратные края.
  3. Пересекаем маски → точная область + чистые края.
  4. Композит на белом фоне.
"""

import logging
from io import BytesIO

import numpy as np
import torch
from PIL import Image, ImageFilter
from rembg import new_session, remove
from scipy import ndimage
from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation

logger = logging.getLogger(__name__)

SEG_MODEL_ID = "sayeed99/segformer_b3_clothes"

# Карта наших категорий в классы SegFormer
CATEGORY_TO_SEG_IDS: dict[str, set[int]] = {
    "верх": {4, 7},         # Upper-clothes, Dress
    "низ": {5, 6},          # Skirt, Pants
    "обувь": {9, 10},       # Left-shoe, Right-shoe
    "аксессуар": {1, 8, 16, 17},  # Hat, Belt, Bag, Scarf
}

# Дилатация маски SegFormer (пикселей)
SEG_DILATION = 8

# Модель rembg
REMBG_MODEL = "u2net"

_seg_model = None
_seg_processor = None
_rembg_session = None


def _get_seg():
    global _seg_model, _seg_processor
    if _seg_model is None:
        logger.info("Загружаем SegFormer для product photo...")
        _seg_processor = AutoImageProcessor.from_pretrained(SEG_MODEL_ID)
        _seg_model = AutoModelForSemanticSegmentation.from_pretrained(SEG_MODEL_ID)
        _seg_model.eval()
        logger.info("SegFormer загружен")
    return _seg_model, _seg_processor


def _get_rembg():
    global _rembg_session
    if _rembg_session is None:
        logger.info("Загружаем rembg %s...", REMBG_MODEL)
        _rembg_session = new_session(REMBG_MODEL)
        logger.info("rembg загружен")
    return _rembg_session


def _seg_mask(image: Image.Image, category: str) -> np.ndarray | None:
    """Маска вещи по категории через SegFormer.

    Возвращает bool-массив (H, W) или None, если категории нет в карте
    или маска пустая.
    """
    target_ids = CATEGORY_TO_SEG_IDS.get(category)
    if not target_ids:
        logger.info("Категория %r не найдена в карте, пропускаем SegFormer", category)
        return None

    model, processor = _get_seg()
    w, h = image.size

    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        logits = torch.nn.functional.interpolate(
            outputs.logits, size=(h, w), mode="bilinear", align_corners=False,
        )
        label_map = logits.argmax(dim=1)[0].cpu().numpy()

    mask = np.isin(label_map, list(target_ids))
    if mask.sum() < 100:
        logger.info("SegFormer: маска пустая (%d пикселей)", mask.sum())
        return None

    mask = ndimage.binary_fill_holes(mask)
    mask = ndimage.binary_dilation(mask, iterations=SEG_DILATION)
    logger.info("SegFormer: маска %.1f%% кадра", mask.sum() / (w * h) * 100)
    return mask


def _rembg_mask(image_bytes: bytes) -> np.ndarray | None:
    """Маска из альфа-канала rembg."""
    try:
        session = _get_rembg()
        rgba_bytes = remove(image_bytes, session=session)
        rgba = Image.open(BytesIO(rgba_bytes))
        alpha = np.array(rgba.split()[-1]) / 255.0
        return alpha > 0.5
    except Exception:
        logger.exception("rembg упал")
        return None


def make_product_photo(image_bytes: bytes, category: str) -> bytes:
    """Сделать товарное фото: только вещь нужной категории на белом фоне.

    Возвращает JPEG-байты. При ошибке — исходные байты (фолбэк).
    """
    try:
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        w, h = image.size

        # 1. Маска SegFormer (по категории)
        mask_seg = _seg_mask(image, category)

        # 2. Маска rembg (аккуратные края)
        mask_rembg = _rembg_mask(image_bytes)

        # 3. Комбинируем
        if mask_seg is not None and mask_rembg is not None:
            # Пересечение: rembg-точность + границы SegFormer
            final_mask = mask_rembg & mask_seg
            if final_mask.sum() < 100:
                logger.info("Пересечение пустое, берём маску SegFormer")
                final_mask = mask_seg
        elif mask_seg is not None:
            final_mask = mask_seg
        elif mask_rembg is not None:
            final_mask = mask_rembg
        else:
            logger.info("Ни одна маска не сработала, отдаю оригинал")
            return image_bytes

        # 4. Сглаживание краёв
        mask_img = Image.fromarray((final_mask * 255).astype(np.uint8), mode="L")
        mask_img = mask_img.filter(ImageFilter.GaussianBlur(radius=1.2))

        # 5. Композит на белом
        white = Image.new("RGB", (w, h), (255, 255, 255))
        white.paste(image, mask=mask_img)

        out = BytesIO()
        white.save(out, format="JPEG", quality=92)
        logger.info(
            "Товарное фото готово: category=%s, %d байт",
            category, len(out.getvalue()),
        )
        return out.getvalue()

    except Exception:
        logger.exception("Ошибка генерации товарного фото")
        return image_bytes


def warmup_product_photo() -> None:
    """Прогреть модели product photo (SegFormer переиспользуется из segmentation)."""
    _get_seg()
    _get_rembg()
