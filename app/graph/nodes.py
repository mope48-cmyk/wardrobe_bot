"""Узлы графа LangGraph."""

import logging

from langchain_core.messages import AIMessage

from app.db.repository import add_item
from app.graph.state import BotState
from app.services.vision import classify_clothing

logger = logging.getLogger(__name__)


WELCOME_TEXT = (
    "Привет! Я помогу каталогизировать твой гардероб "
    "и подбирать одежду по погоде.\n\n"
    "Доступные команды:\n"
    "/add — добавить вещь\n"
    "/list — показать гардероб\n"
    "/outfit — подобрать комплект по погоде\n"
    "/help — справка\n"
    "/cancel — отменить текущее действие"
)

HELP_TEXT = (
    "Что я умею:\n\n"
    "• /add — добавить вещь. Пришлите фото, я попробую распознать категорию и тип.\n\n"
    "• /list — показать все вещи в вашем гардеробе.\n\n"
    "• /outfit — подобрать комплект по погоде.\n\n"
    "• /cancel — отменить текущий диалог."
)


async def start_node(state: BotState) -> dict:
    logger.info("start_node: user_id=%s", state.get("user_id"))
    return {
        "messages": [AIMessage(content=WELCOME_TEXT)],
        "intent": None, "step": None, "draft_item": None,
    }


async def help_node(state: BotState) -> dict:
    return {"messages": [AIMessage(content=HELP_TEXT)]}


async def cancel_node(state: BotState) -> dict:
    return {
        "messages": [AIMessage(content="Хорошо, отменил. Что делаем дальше?")],
        "intent": None, "step": None, "draft_item": None,
    }


async def fallback_node(state: BotState) -> dict:
    return {"messages": [AIMessage(content="Не понял. Наберите /help.")]}


async def stub_node(state: BotState) -> dict:
    return {"messages": [AIMessage(content="Эта команда появится позже.")]}


# ---------- /add ----------

VALID_CATEGORIES = {"верх", "низ", "обувь", "аксессуар"}
VALID_FORMAL = {"casual", "business", "sport"}
VALID_SEASON = {"лето", "демисезон", "зима", "универсальная"}

STEP_TO_FIELD = {
    "awaiting_category": "category",
    "awaiting_type": "type",
    "awaiting_color": "color",
    "awaiting_warmth": "warmth_level",
    "awaiting_waterproof": "waterproof",
    "awaiting_formal": "formal_level",
    "awaiting_season": "season",
}

NEXT_STEP = {
    "awaiting_category": (
        "awaiting_type",
        "Уточните тип вещи (например: футболка, джинсы, куртка, кроссовки):",
    ),
    "awaiting_type": ("awaiting_color", "Какого цвета вещь?"),
    "awaiting_color": (
        "awaiting_warmth",
        "Насколько вещь тёплая? Оцените от 1 (очень лёгкая) до 5 (очень тёплая):",
    ),
    "awaiting_warmth": (
        "awaiting_waterproof",
        "Вещь водонепроницаемая? Напишите да или нет:",
    ),
    "awaiting_waterproof": (
        "awaiting_formal",
        "Для какого случая вещь? Варианты: casual, business, sport:",
    ),
    "awaiting_formal": (
        "awaiting_season",
        "Для какого сезона? Варианты: лето, демисезон, зима, универсальная:",
    ),
}


def _validate_answer(step: str, text: str) -> tuple[bool, object, str]:
    text = text.strip()
    lower = text.lower()

    if step == "awaiting_category":
        if lower in VALID_CATEGORIES:
            return True, lower, ""
        return False, None, "Выберите: верх, низ, обувь или аксессуар."

    if step == "awaiting_type":
        if len(text) >= 2:
            return True, text, ""
        return False, None, "Напишите название типа вещи."

    if step == "awaiting_color":
        if len(text) >= 2:
            return True, text, ""
        return False, None, "Напишите цвет вещи текстом."

    if step == "awaiting_warmth":
        try:
            n = int(text)
            if 1 <= n <= 5:
                return True, n, ""
        except ValueError:
            pass
        return False, None, "Введите число от 1 до 5."

    if step == "awaiting_waterproof":
        if lower in ("да", "yes", "true", "1"):
            return True, True, ""
        if lower in ("нет", "no", "false", "0"):
            return True, False, ""
        return False, None, "Ответьте да или нет."

    if step == "awaiting_formal":
        if lower in VALID_FORMAL:
            return True, lower, ""
        return False, None, "Выберите: casual, business или sport."

    if step == "awaiting_season":
        if lower in VALID_SEASON:
            return True, lower, ""
        return False, None, "Выберите: лето, демисезон, зима или универсальная."

    return False, None, "Неизвестный шаг."


async def add_start_node(state: BotState) -> dict:
    return {
        "messages": [AIMessage(content="Пришлите, пожалуйста, фотографию вещи.")],
        "intent": "add", "step": "awaiting_photo", "draft_item": {},
    }


async def add_expect_photo_node(state: BotState) -> dict:
    return {
        "messages": [
            AIMessage(content="Жду фотографию вещи. Или /cancel для отмены.")
        ],
    }


async def add_photo_node(state: BotState) -> dict:
    """Фото получено: распознаём категорию, тип и цвет."""
    file_id = state.get("input_photo_file_id")
    image_bytes = state.get("input_photo_bytes")

    draft = dict(state.get("draft_item") or {})
    draft["photo_file_id"] = file_id

    if not image_bytes:
        return {
            "messages": [
                AIMessage(
                    content=(
                        "Не удалось обработать фото.\n\n"
                        "Что это за вещь? Напишите: <b>верх</b>, <b>низ</b>, "
                        "<b>обувь</b> или <b>аксессуар</b>."
                    )
                )
            ],
            "step": "awaiting_category",
            "draft_item": draft,
        }

    result = await classify_clothing(image_bytes)

    # Цвет всегда есть — даже если CLIP не смог распознать вещь
    draft["predicted_color"] = result.get("color")

    if not result.get("ok") or result.get("category") == "другое":
        return {
            "messages": [
                AIMessage(
                    content=(
                        "Не смог распознать вещь на фото.\n\n"
                        "Что это за вещь? Напишите: <b>верх</b>, <b>низ</b>, "
                        "<b>обувь</b> или <b>аксессуар</b>."
                    )
                )
            ],
            "step": "awaiting_category",
            "draft_item": draft,
        }

    draft["predicted_category"] = result["category"]
    draft["predicted_type"] = result["type"]
    draft["confidence"] = result["confidence"]

    confidence_pct = int(result["confidence"] * 100)
    return {
        "messages": [
            AIMessage(
                content=(
                    f"Похоже, это <b>{result['type']}</b> "
                    f"(категория: <b>{result['category']}</b>, "
                    f"цвет: <b>{result['color']}</b>, "
                    f"уверенность: {confidence_pct}%).\n\n"
                    "Всё верно? Ответьте <b>да</b> или <b>нет</b>."
                )
            )
        ],
        "step": "awaiting_confirm_category",
        "draft_item": draft,
    }

    # Распознавание
    result = await classify_clothing(image_bytes)

    if not result.get("ok") or result["category"] == "другое":
        # Распознавание не удалось — ручной ввод
        return {
            "messages": [
                AIMessage(
                    content=(
                        "Не смог распознать вещь на фото.\n\n"
                        "Что это за вещь? Напишите: <b>верх</b>, <b>низ</b>, "
                        "<b>обувь</b> или <b>аксессуар</b>."
                    )
                )
            ],
            "step": "awaiting_category",
            "draft_item": draft,
        }

    # Сохраняем предсказание в черновик
    draft["predicted_category"] = result["category"]
    draft["predicted_type"] = result["type"]
    draft["confidence"] = result["confidence"]

    confidence_pct = int(result["confidence"] * 100)
    return {
        "messages": [
            AIMessage(
                content=(
                    f"Похоже, это <b>{result['type']}</b> "
                    f"(категория: <b>{result['category']}</b>, "
                    f"уверенность: {confidence_pct}%).\n\n"
                    "Всё верно? Ответьте <b>да</b> или <b>нет</b>."
                )
            )
        ],
        "step": "awaiting_confirm_category",
        "draft_item": draft,
    }


async def add_confirm_category_node(state: BotState) -> dict:
    """Обработка подтверждения предсказания CLIP (категория, тип, цвет)."""
    text = (state.get("input_text") or "").strip().lower()
    draft = dict(state.get("draft_item") or {})

    if text in ("да", "yes", "верно", "ага", "ok"):
        draft["category"] = draft["predicted_category"]
        draft["type"] = draft["predicted_type"]
        draft["color"] = draft.get("predicted_color") or ""

        return {
            "messages": [
                AIMessage(
                    content=(
                        "Отлично!\n\n"
                        "Насколько вещь тёплая? Оцените от 1 (очень лёгкая) "
                        "до 5 (очень тёплая):"
                    )
                )
            ],
            "step": "awaiting_warmth",
            "draft_item": draft,
        }

    if text in ("нет", "no", "неверно"):
        return {
            "messages": [
                AIMessage(
                    content=(
                        "Хорошо. Напишите категорию: <b>верх</b>, <b>низ</b>, "
                        "<b>обувь</b> или <b>аксессуар</b>."
                    )
                )
            ],
            "step": "awaiting_category",
            "draft_item": draft,
        }

    return {
        "messages": [AIMessage(content="Ответьте <b>да</b> или <b>нет</b>.")],
    }


async def add_attribute_node(state: BotState) -> dict:
    step = state.get("step")
    text = state.get("input_text") or ""

    logger.info("add_attribute_node: step=%s text=%r", step, text)

    valid, value, error = _validate_answer(step, text)
    if not valid:
        return {"messages": [AIMessage(content=error)]}

    draft = dict(state.get("draft_item") or {})
    draft[STEP_TO_FIELD[step]] = value

    # Последний шаг — сохраняем
    if step == "awaiting_season":
        try:
            await add_item(
                user_id=state["user_id"],
                photo_file_id=draft["photo_file_id"],
                category=draft["category"],
                type=draft["type"],
                color=draft["color"],
                material="",
                warmth_level=draft["warmth_level"],
                waterproof=draft["waterproof"],
                formal_level=draft["formal_level"],
                season=draft["season"],
            )
        except Exception as e:
            logger.exception("Ошибка сохранения вещи")
            return {
                "messages": [AIMessage(content=f"Не удалось сохранить: {e}")],
                "step": None, "draft_item": None, "intent": None,
            }

        return {
            "messages": [
                AIMessage(
                    content=(
                        "Готово! Сохранил вещь:\n"
                        f"• {draft['category']} / {draft['type']}\n"
                        f"• цвет: {draft['color']}\n"
                        f"• тепло: {draft['warmth_level']}/5\n"
                        f"• водонепроницаемая: "
                        f"{'да' if draft['waterproof'] else 'нет'}\n"
                        f"• стиль: {draft['formal_level']}\n"
                        f"• сезон: {draft['season']}\n\n"
                        "Добавьте ещё вещь командой /add."
                    )
                )
            ],
            "step": None, "draft_item": None, "intent": None,
        }

    next_step, question = NEXT_STEP[step]
    return {
        "messages": [AIMessage(content=question)],
        "step": next_step,
        "draft_item": draft,
    }