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


def test_canon_color():
    from tools.eval_recognition import canon_color

    assert canon_color("черный") == "чёрный"
    assert canon_color("чёрный") == "чёрный"
    assert canon_color("темно-синий") == "тёмно-синий"
    assert canon_color("голубые") == "голубой"
    assert canon_color("синий") == "синий"


def test_collect_labels_uses_manifest(tmp_path):
    """При наличии manifest.csv метки берутся из него, а не из имени файла."""
    from tools.eval_recognition import collect_labels

    folder = tmp_path / "ds"
    folder.mkdir()
    (folder / "0005.jpg").write_bytes(b"x")
    (folder / "0006.jpg").write_bytes(b"x")
    (folder / "manifest.csv").write_text(
        "file,category,type,color\n"
        "0005.jpg,обувь,ботинки,тёмно-коричневый\n"
        "0006.jpg,верх,толстовка с капюшоном,синий\n",
        encoding="utf-8",
    )

    files, labels = collect_labels(folder)

    assert len(files) == 2
    assert labels[0] == {
        "category": "обувь",
        "type": "ботинки",
        "color": "тёмно-коричневый",
    }
    # Многословный тип сохраняется целиком
    assert labels[1]["type"] == "толстовка с капюшоном"
    assert labels[1]["color"] == "синий"

