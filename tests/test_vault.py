import os

import pytest
from cryptography.fernet import Fernet

from pd_shield.normalize import normalize_person
from pd_shield.vault import Vault, VaultError


@pytest.fixture
def key():
    return Fernet.generate_key()


@pytest.fixture
def vault(tmp_path, key):
    return Vault(str(tmp_path / "v.enc"), key)


def test_детерминированность_меток_для_форм_одного_человека(vault):
    forms = ["Иванова Мария", "Ивановой Марии", "Иванова М.П.",
             "М. П. Иванова", "с Марией Ивановой"]
    labels = {vault.label_for_person(normalize_person(f)) for f in forms}
    assert labels == {"PERSON_1"}


def test_разные_люди_разные_метки(vault):
    a = vault.label_for_person(normalize_person("Иванова Мария"))
    b = vault.label_for_person(normalize_person("Иванова Анна"))
    c = vault.label_for_person(normalize_person("Петров Иван"))
    assert len({a, b, c}) == 3


def test_инициалы_дополняются_полным_именем(vault):
    lab1 = vault.label_for_person(normalize_person("Иванова М.П."))
    lab2 = vault.label_for_person(normalize_person("Иванова Мария Петровна"))
    assert lab1 == lab2
    kind, entry = vault.lookup(lab1)
    assert kind == "person"
    assert entry["first"] == "Мария"
    assert entry["middle"] == "Петровна"


def test_метки_переживают_перезагрузку(tmp_path, key):
    path = str(tmp_path / "v.enc")
    v1 = Vault(path, key)
    lab_p = v1.label_for_person(normalize_person("Иванова Мария"))
    lab_t = v1.label_for_value("phone", "79001234567", "+7 900 123-45-67")
    v1.save()

    v2 = Vault(path, key)
    assert v2.label_for_person(normalize_person("Ивановой Марии")) == lab_p
    assert v2.label_for_value("phone", "79001234567", "x") == lab_t
    # счётчик продолжается, а не начинается заново
    assert v2.label_for_person(normalize_person("Петров Иван")) == "PERSON_2"


def test_без_ключа_файл_не_читается(tmp_path, key):
    path = str(tmp_path / "v.enc")
    v = Vault(path, key)
    v.label_for_person(normalize_person("Иванова Мария"))
    v.save()

    # файл на диске не содержит открытых данных
    blob = open(path, "rb").read()
    assert "Иванова".encode() not in blob
    assert b"PERSON_1" not in blob

    # с чужим ключом — понятная ошибка, а не мусор
    with pytest.raises(VaultError):
        Vault(path, Fernet.generate_key())


def test_кривой_ключ_даёт_понятную_ошибку(tmp_path):
    with pytest.raises(VaultError):
        Vault(str(tmp_path / "v.enc"), b"prosto-parol")


def test_значения_детерминированы_по_нормализованному_ключу(vault):
    a = vault.label_for_value("phone", "79001234567", "8 900 123 45 67")
    b = vault.label_for_value("phone", "79001234567", "+79001234567")
    assert a == b == "PHONE_1"


def test_lookup_неизвестной_метки(vault):
    assert vault.lookup("PERSON_99") is None


def test_второй_экземпляр_видит_чужие_записи_без_перезапуска(tmp_path, key):
    """Сценарий бота: vault ответов создан при старте, индексация пишет
    метки позже. Restorer обязан их видеть без пересоздания объекта
    (баг «[неизвестно]» вместо имён, боевой проект, 02.08.2026)."""
    path = str(tmp_path / "v.enc")
    answerer_vault = Vault(path, key)          # создан при «старте бота»
    ingest_vault = Vault(path, key)            # «индексация» в том же процессе

    label = ingest_vault.label_for_person(normalize_person("Иванова Мария"))
    ingest_vault.save()
    os.utime(path, (0, 0))  # гарантированно другой mtime, чем при создании

    found = answerer_vault.lookup(label)
    assert found is not None
    assert found[0] == "person"
    # и счётчик после перечитывания продолжается, а не сталкивается
    assert (answerer_vault.label_for_person(normalize_person("Петров Иван"))
            == "PERSON_2")
