"""Санитайзинг HTML для сообщений бота.

Ответы LLM (GigaChat) могут прийти с некорректным или злонамеренным HTML.
aiogram отправляет текст с parse_mode="HTML"; если теги битые или
неизвестные — Telegram отвечает 400 «can't parse entities» и сообщение
теряется. sanitize_html оставляет только безопасный поднабор тегов,
всё остальное экранирует.
"""

import re
from html import escape

_ALLOWED_TAGS = {"b", "i", "u", "s", "br", "code"}

# <tag attr="..."> | </tag> | <tag/>
_TAG_RE = re.compile(
    r"<(/?)([A-Za-z][A-Za-z0-9]*)((?:[^<>\"']|\"[^\"]*\"|'[^']*')*?)(/?)>"
)


def _safe_tag(m: re.Match) -> str:
    closing, tag, _, selfclose = m.groups()
    tag_lower = tag.lower()

    # Закрывающие теги сохраняем только для разрешённых имён
    if closing:
        if tag_lower in _ALLOWED_TAGS:
            return f"</{tag_lower}>"
        return escape(m.group(0))

    if tag_lower not in _ALLOWED_TAGS:
        return escape(m.group(0))

    return f"<{closing}{tag_lower}{selfclose}>"


def sanitize_html(text: str | None) -> str:
    """Экранировать небезопасный HTML, сохранив разрешённые теги."""
    if not text:
        return ""

    parts = []
    pos = 0
    for m in _TAG_RE.finditer(text):
        parts.append(escape(text[pos:m.start()]))
        parts.append(_safe_tag(m))
        pos = m.end()
    parts.append(escape(text[pos:]))
    return "".join(parts)
