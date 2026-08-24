import json

import pytest

from pd_shield.config import ShieldConfig


@pytest.fixture(autouse=True)
def чистое_окружение(monkeypatch):
    """Словарь не должен протекать из окружения машины в тесты."""
    monkeypatch.delenv("PD_NAMES", raising=False)


def test_словарь_читается_из_переменной_окружения(monkeypatch):
    monkeypatch.setenv("PD_NAMES", '["Иванова Мария", "Петров Иван"]')
    assert ShieldConfig().names == ["Иванова Мария", "Петров Иван"]


def test_словарь_из_json_имеет_приоритет_над_переменной(monkeypatch):
    monkeypatch.setenv("PD_NAMES", '["Из переменной"]')
    cfg = ShieldConfig(names=["Из конфига"])
    assert cfg.names == ["Из конфига"]


def test_без_переменной_словарь_пустой():
    assert ShieldConfig().names == []


def test_пустая_переменная_даёт_пустой_словарь(monkeypatch):
    monkeypatch.setenv("PD_NAMES", "")
    assert ShieldConfig().names == []


def test_битый_json_в_переменной_даёт_понятную_ошибку(monkeypatch):
    monkeypatch.setenv("PD_NAMES", "Иванова, Петров")
    with pytest.raises(ValueError, match="PD_NAMES содержит не JSON"):
        ShieldConfig()


def test_не_список_в_переменной_даёт_понятную_ошибку(monkeypatch):
    monkeypatch.setenv("PD_NAMES", '{"a": 1}')
    with pytest.raises(ValueError, match="ожидается список строк"):
        ShieldConfig()


def test_список_не_строк_в_переменной_даёт_понятную_ошибку(monkeypatch):
    monkeypatch.setenv("PD_NAMES", "[1, 2, 3]")
    with pytest.raises(ValueError, match="ожидается список строк"):
        ShieldConfig()


def test_имя_переменной_настраивается(monkeypatch):
    monkeypatch.setenv("PD_NAMES_ДРУГОЙ_БОТ", '["Бахметьева Олеся"]')
    cfg = ShieldConfig(names_env="PD_NAMES_ДРУГОЙ_БОТ")
    assert cfg.names == ["Бахметьева Олеся"]


def test_from_json_принимает_names_env(tmp_path, monkeypatch):
    monkeypatch.setenv("PD_NAMES", '["Иванова Мария"]')
    path = tmp_path / "pd_config.json"
    path.write_text(json.dumps({
        "names_env": "PD_NAMES",
        "enabled_types": ["person", "phone"],
        "vault_path": "/data/pd_vault.enc",
        "key_env": "PD_SHIELD_KEY",
    }, ensure_ascii=False), encoding="utf-8")
    cfg = ShieldConfig.from_json(str(path))
    assert cfg.names == ["Иванова Мария"]
    assert cfg.enabled_types == ["person", "phone"]


def test_from_json_отвергает_неизвестное_поле(tmp_path):
    path = tmp_path / "pd_config.json"
    path.write_text(json.dumps({"неизвестное": 1}), encoding="utf-8")
    with pytest.raises(ValueError, match="Неизвестные поля"):
        ShieldConfig.from_json(str(path))
