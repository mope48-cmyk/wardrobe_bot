"""Тесты fallback-цепи LLM: GigaChat → OpenRouter → шаблон."""

from app.services import llm_stylist as M


class _FakeGigaChatResponse:
    def __init__(self, content):
        self.content = content


async def _make_item(make_item):
    return make_item(category="верх", type="футболка", colors='["белый"]')


# ---------------------------------------------------------------------------
# _llm_generate: выбор провайдера
# ---------------------------------------------------------------------------


async def test_llm_generate_uses_gigachat_first(monkeypatch):
    class FakeLLM:
        @staticmethod
        async def ainvoke(messages):
            return _FakeGigaChatResponse("  ответ гигачата  ")

    calls = {"openrouter": 0}

    async def fake_openrouter(system, user, temperature=0.7):
        calls["openrouter"] += 1
        return "ответ openrouter"

    monkeypatch.setattr(M, "_llm", FakeLLM())
    monkeypatch.setattr(M, "_openrouter_complete", fake_openrouter)

    result = await M._llm_generate("sys", "user", temperature=0.5)
    assert result == "ответ гигачата"
    assert calls["openrouter"] == 0


async def test_llm_generate_falls_back_to_openrouter_when_gigachat_fails(monkeypatch):
    class BrokenLLM:
        @staticmethod
        async def ainvoke(messages):
            raise RuntimeError("GigaChat down")

    async def fake_openrouter(system, user, temperature=0.7):
        return "ответ openrouter"

    monkeypatch.setattr(M, "_llm", BrokenLLM())
    monkeypatch.setattr(M, "_openrouter_complete", fake_openrouter)

    result = await M._llm_generate("sys", "user")
    assert result == "ответ openrouter"


async def test_llm_generate_returns_none_when_both_unavailable(monkeypatch):
    class BrokenLLM:
        @staticmethod
        async def ainvoke(messages):
            raise RuntimeError("GigaChat down")

    async def fake_openrouter(system, user, temperature=0.7):
        return None

    monkeypatch.setattr(M, "_llm", BrokenLLM())
    monkeypatch.setattr(M, "_openrouter_complete", fake_openrouter)

    assert await M._llm_generate("sys", "user") is None


# ---------------------------------------------------------------------------
# Публичные генераторы
# ---------------------------------------------------------------------------


async def test_generate_vasiliev_comment_fallback_to_openrouter(monkeypatch, make_item):
    items = [await _make_item(make_item)]

    async def fake_openrouter(system, user, temperature=0.7):
        return "Браво! Прекрасный выбор на сегодня."

    monkeypatch.setattr(M, "_llm", None)
    monkeypatch.setattr(M, "_openrouter_complete", fake_openrouter)

    weather = {"feels_like": 15, "description": "ясно"}
    result = await M.generate_vasiliev_comment(
        weather, "Москва", "прогулка", items, max_chars=200
    )
    assert "Браво!" in result


async def test_generate_vasiliev_comment_returns_none_without_llm(monkeypatch, make_item):
    items = [await _make_item(make_item)]

    async def fake_openrouter(system, user, temperature=0.7):
        return None

    monkeypatch.setattr(M, "_llm", None)
    monkeypatch.setattr(M, "_openrouter_complete", fake_openrouter)

    result = await M.generate_vasiliev_comment(
        {"feels_like": 15}, "Москва", "прогулка", items
    )
    assert result is None


async def test_generate_chat_response_template_fallback(monkeypatch):
    async def fake_openrouter(system, user, temperature=0.7):
        return None

    monkeypatch.setattr(M, "_llm", None)
    monkeypatch.setattr(M, "_openrouter_complete", fake_openrouter)

    result = await M.generate_chat_response("привет")
    assert result is not None
    assert "понимаю" in result or "info" in result
