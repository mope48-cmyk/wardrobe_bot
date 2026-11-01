"""Тесты парсинга имён датасета распознавания (tools/eval_recognition.py)."""

from tools.eval_recognition import parse_filename


def test_parse_filename_full():
    assert parse_filename("05_Верх_рубашка_синий_на_белом.jpg") == {
        "category": "верх",
        "type": "рубашка",
        "color": "синий",
    }


def test_parse_filename_no_color():
    # «не нормирован» — это не цвет, цвет пропускается.
    assert parse_filename("07_Верх_свитер_не_нормирован_кашемир.jpg") == {
        "category": "верх",
        "type": "свитер",
        "color": None,
    }


def test_parse_filename_shoes_and_jpg():
    assert parse_filename("33_Обувь_ботинки_коричневый_на_полу.jpg") == {
        "category": "обувь",
        "type": "ботинки",
        "color": "коричневый",
    }


def test_parse_filename_accessories():
    assert parse_filename("39_Аксессуары_сумка_чёрный_на_черном.jpg") == {
        "category": "аксессуар",
        "type": "сумка",
        "color": "чёрный",
    }


def test_parse_filename_short_rejected():
    assert parse_filename("01_Верх.jpg") is None
    assert parse_filename("whatever.jpg") is None
