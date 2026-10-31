"""Общие фикстуры и стубы тяжёлых ML-модулей.

В venv разработчика нет torch/transformers/rembg/scipy (это тяжёлые
зависимости, нужные только на сервере). Чтобы можно было импортировать
app.graph.nodes / app.services.* и гонять юнит-тесты на обычной машине,
мы подменяем эти модули лёгкими заглушками ДО импорта приложения.
"""

import os
import sys
import tempfile
import types
from pathlib import Path

# ---------------------------------------------------------------------------
# 1. Окружение: ставим настройки ДО импорта app.config (там read-only Settings)
# ---------------------------------------------------------------------------
_TMP_DB_DIR = tempfile.mkdtemp(prefix="wardrobe_test_")

os.environ.setdefault("BOT_TOKEN", "test-token")
os.environ.setdefault("TELEGRAM_API_URL", "")
os.environ.setdefault("HF_TOKEN", "")
os.environ.setdefault("OWM_API_KEY", "test-owm")
os.environ.setdefault("GROQ_API_KEY", "")
os.environ.setdefault("OPENROUTER_API_KEY", "")
os.environ.setdefault("GIGACHAT_CREDENTIALS", "")
os.environ["DATABASE_URL"] = (
    f"sqlite+aiosqlite:///{Path(_TMP_DB_DIR).as_posix()}/test.db"
)

# ---------------------------------------------------------------------------
# 2. Стубы тяжёлых ML-модулей
# ---------------------------------------------------------------------------


def _make_stub(name: str, attrs: tuple[str, ...] = (), funcs: tuple[str, ...] = ()):
    mod = types.ModuleType(name)
    for attr in attrs:
        setattr(mod, attr, type(attr, (), {}))
    for fn in funcs:
        setattr(mod, fn, lambda *a, **k: None)
    sys.modules[name] = mod
    return mod


_make_stub("rembg", ("new_session", "remove"))
_make_stub(
    "transformers",
    (
        "AutoImageProcessor", "AutoModelForSemanticSegmentation", "AutoModel",
        "AutoProcessor", "AutoTokenizer", "pipeline",
        "SegformerForImageClassification", "SegformerImageProcessor",
        "CLIPModel", "CLIPProcessor",
    ),
)
_make_stub(
    "torch",
    ("no_grad", "Tensor", "device", "load", "save", "from_numpy", "float32"),
    ("set_num_threads",),
)
_make_stub("scipy")
_make_stub("scipy.ndimage")
_make_stub("onnxruntime")

# ---------------------------------------------------------------------------
# 3. Общие фикстуры
# ---------------------------------------------------------------------------

import pytest  # noqa: E402


@pytest.fixture
async def db():
    """Инициализирует временную БД и чистит таблицы после теста."""
    from app.db.database import SessionLocal, engine, init_db

    await init_db()
    yield engine, SessionLocal

    async with engine.begin() as conn:
        from sqlalchemy import text
        await conn.execute(text("DELETE FROM wardrobe_items"))
        await conn.execute(text("DELETE FROM user_morning_settings"))
        await conn.execute(text("DELETE FROM user_preferences"))
        await conn.commit()


@pytest.fixture
def make_item():
    """Фабрика фейковых вещей без обращения к БД."""
    from types import SimpleNamespace

    def _make(**overrides):
        base = {
            "id": 0,
            "user_id": 1,
            "photo_file_id": "f0",
            "generated_photo_file_id": None,
            "category": "верх",
            "type": "футболка",
            "colors": "[]",
            "material": "",
            "warmth_level": 3,
            "waterproof": False,
            "thickness": 3,
            "has_hood": False,
            "use_white_overlay": False,
            "formal_level": "casual",
            "season": "универсальная",
            "layer": "base",
            "accessory_occasions": "[]",
            "is_in_laundry": False,
            "laundry_start_date": None,
        }
        base.update(overrides)
        return SimpleNamespace(**base)

    return _make
