"""Тесты рассылки релиза: разбиение длинного текста и маркер отправки."""

import hashlib
import re

import pytest

from app.services.broadcast import MAX_MESSAGE_CHARS, _split_message, send_release_broadcast


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def _fake_user_ids(user_ids):
    async def get():
        return user_ids
    return get


def test_split_message_chunks_within_limit():
    """Длинный текст релиза должен разбиваться на сообщения <= лимита."""
    blocks = [
        f"<b>Блок {i}</b>\n" + f"Абзац номер {i}. " * 80
        for i in range(12)
    ]
    text = "\n\n".join(blocks)
    assert len(text) > MAX_MESSAGE_CHARS

    chunks = _split_message(text)
    assert len(chunks) > 1
    for c in chunks:
        assert len(c) <= MAX_MESSAGE_CHARS
        # Каждый чанк — валидный HTML: теги сбалансированы.
        assert c.count("<b>") == c.count("</b>")
    # Всё содержимое сохранилось (сравнение без учёта пробельных склейек).
    assert _norm("\n\n".join(chunks)) == _norm(text)


def test_split_message_short_text_single_chunk():
    text = "<b>Короткий релиз.</b>"
    assert _split_message(text) == [text]


class FakeBot:
    def __init__(self):
        self.sent = []
        self.fail_with = None

    async def send_message(self, chat_id, text, parse_mode=None):
        if self.fail_with is not None:
            raise self.fail_with
        self.sent.append((chat_id, text, parse_mode))

    async def close(self):
        pass


async def _run_broadcast(tmp_path, bot, user_ids, notes_text):
    notes = tmp_path / "release_notes.txt"
    notes.write_text(notes_text, encoding="utf-8")
    marker = tmp_path / "release_notes.sent"
    return await send_release_broadcast(
        bot, file_path=str(notes), sent_marker=str(marker)
    ), marker


@pytest.mark.asyncio
async def test_broadcast_success_writes_sentinel(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "app.services.broadcast.get_all_user_ids", _fake_user_ids([111, 222])
    )
    bot = FakeBot()
    text = "<b>Релиз.</b>\n\n<b>Изменения.</b>"
    _, marker = await _run_broadcast(tmp_path, bot, [111, 222], text)

    assert marker.exists()
    assert marker.read_text(encoding="utf-8") == text
    # Короткий текст уходит одним сообщением каждому пользователю.
    assert bot.sent == [(111, text, "HTML"), (222, text, "HTML")]


@pytest.mark.asyncio
async def test_broadcast_total_failure_keeps_no_sentinel(tmp_path, monkeypatch):
    from aiogram.exceptions import TelegramBadRequest

    monkeypatch.setattr(
        "app.services.broadcast.get_all_user_ids", _fake_user_ids([111])
    )
    bot = FakeBot()
    bot.fail_with = TelegramBadRequest(
        method="sendMessage", message="Bad Request: message is too long"
    )
    _, marker = await _run_broadcast(tmp_path, bot, [111], "<b>Релиз.</b>")

    # Ни одного успеха — маркер НЕ создаётся, рассылка повторится.
    assert not marker.exists()
    assert bot.sent == []


@pytest.mark.asyncio
async def test_broadcast_long_text_sends_chunks(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "app.services.broadcast.get_all_user_ids", _fake_user_ids([111])
    )
    bot = FakeBot()
    text = "\n\n".join(
        f"<b>Блок {i}</b>\n" + f"Строка {i}. " * 90 for i in range(10)
    )
    _, marker = await _run_broadcast(tmp_path, bot, [111], text)

    assert len(bot.sent) > 1
    for chat_id, msg, _parse in bot.sent:
        assert chat_id == 111
        assert len(msg) <= MAX_MESSAGE_CHARS
        assert msg.count("<b>") == msg.count("</b>")
    assert marker.exists()


@pytest.mark.asyncio
async def test_broadcast_incremental_sends_only_new(tmp_path, monkeypatch):
    """Дописанный блок уходит отдельно, повторный запуск ничего не шлёт."""
    monkeypatch.setattr(
        "app.services.broadcast.get_all_user_ids", _fake_user_ids([111])
    )
    bot = FakeBot()
    notes = tmp_path / "release_notes.txt"
    marker = tmp_path / "release_notes.sent"

    first = "<b>Релиз 1.</b>"
    notes.write_text(first, encoding="utf-8")
    await send_release_broadcast(bot, file_path=str(notes), sent_marker=str(marker))
    assert [c[1] for c in bot.sent] == [first]
    assert marker.read_text(encoding="utf-8") == first

    # Повторный запуск без изменений — ничего не шлём.
    bot.sent.clear()
    await send_release_broadcast(bot, file_path=str(notes), sent_marker=str(marker))
    assert bot.sent == []

    # Дописали в конец — уходит только новый блок.
    second = "\n\n<b>Релиз 2.</b>"
    notes.write_text(first + second, encoding="utf-8")
    await send_release_broadcast(bot, file_path=str(notes), sent_marker=str(marker))
    assert [c[1] for c in bot.sent] == ["<b>Релиз 2.</b>"]
    assert marker.read_text(encoding="utf-8") == first + second


@pytest.mark.asyncio
async def test_broadcast_legacy_md5_sentinel(tmp_path, monkeypatch):
    """Старый маркер (md5): совпадение — пропуск, несовпадение — полная рассылка."""
    monkeypatch.setattr(
        "app.services.broadcast.get_all_user_ids", _fake_user_ids([111])
    )
    notes = tmp_path / "release_notes.txt"
    marker = tmp_path / "release_notes.sent"
    text = "<b>Релиз.</b>"
    notes.write_text(text, encoding="utf-8")

    old_hash = hashlib.md5(text.encode("utf-8")).hexdigest()
    marker.write_text(old_hash, encoding="utf-8")
    bot = FakeBot()
    await send_release_broadcast(bot, file_path=str(notes), sent_marker=str(marker))
    assert bot.sent == []

    marker.write_text("0" * 32, encoding="utf-8")
    bot = FakeBot()
    await send_release_broadcast(bot, file_path=str(notes), sent_marker=str(marker))
    assert bot.sent == [(111, text, "HTML")]
    assert marker.read_text(encoding="utf-8") == text
