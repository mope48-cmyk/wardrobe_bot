"""Тесты санитайзинга HTML (app/services/html_sanitizer.py)."""

import pytest

from app.services.html_sanitizer import sanitize_html


@pytest.mark.parametrize(
    "raw,expected",
    [
        # Разрешённые теги сохраняются
        ("<b>жирный</b>", "<b>жирный</b>"),
        ("<i>курсив</i> и <s>зачёркнуто</s>", "<i>курсив</i> и <s>зачёркнуто</s>"),
        ("текст<br>перенос", "текст<br>перенос"),
        # Неизвестные/опасные теги экранируются
        ("<script>alert(1)</script>", "&lt;script&gt;alert(1)&lt;/script&gt;"),
        ("<img src=x onerror=alert(1)>", "&lt;img src=x onerror=alert(1)&gt;"),
        ("<unknown>текст</unknown>", "&lt;unknown&gt;текст&lt;/unknown&gt;"),
        # Ссылки не поддерживаются — любые теги <a> экранируются целиком
        (
            '<a href="https://example.com">ссылка</a>',
            "&lt;a href=&quot;https://example.com&quot;&gt;ссылка&lt;/a&gt;",
        ),
        (
            '<a href="javascript:alert(1)">жми</a>',
            "&lt;a href=&quot;javascript:alert(1)&quot;&gt;жми&lt;/a&gt;",
        ),
        (
            '<a href="file:///etc/passwd">файл</a>',
            "&lt;a href=&quot;file:///etc/passwd&quot;&gt;файл&lt;/a&gt;",
        ),
        # Спецсимволы в тексте экранируются
        ("R&D < 5", "R&amp;D &lt; 5"),
        # Пустые значения
        ("", ""),
        (None, ""),
    ],
)
def test_sanitize_html(raw, expected):
    assert sanitize_html(raw) == expected


def test_sanitize_keeps_entities_from_llm():
    # LLM вернула амперсанд в тексте — он экранируется, но отображается корректно
    assert sanitize_html("куртка & брюки") == "куртка &amp; брюки"
