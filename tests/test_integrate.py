"""Сквозной тест: документ -> маскирование -> «ответ LLM» -> восстановление.

LLM имитируется классом-заглушкой (правило 9 проекта): тесту не нужны
ни сеть, ни ключ Anthropic.
"""

import json

import pytest
from cryptography.fernet import Fernet

from pd_shield import Shield, ShieldConfig, CASE_INSTRUCTION


class _FakeBlock:
    type = "text"

    def __init__(self, text):
        self.text = text


class _FakeResponse:
    def __init__(self, text):
        self.content = [_FakeBlock(text)]


class _FakeClient:
    """Возвращает заранее заданный «ответ» и запоминает, что ему прислали."""

    def __init__(self, reply):
        self._reply = reply
        self.seen = None
        self.messages = self

    def create(self, **kwargs):
        self.seen = kwargs
        return _FakeResponse(self._reply)


@pytest.fixture
def shield(tmp_path):
    cfg = ShieldConfig(
        names=["Иванова Мария Петровна"],
        vault_path=str(tmp_path / "v.enc"),
    )
    return Shield(cfg, key=Fernet.generate_key())


def test_сквозной_обезличить_и_восстановить(shield):
    doc = ("Классный руководитель Иванова Мария Петровна, "
           "телефон 8 (900) 123-45-67. Замена: Петров Семён Ильич.")

    masked = shield.mask_for_index(doc)
    assert "Иванова" not in masked and "Петров" not in masked
    assert "123-45-67" not in masked
    assert "[PERSON_1]" in masked and "[PERSON_2]" in masked
    assert "[PHONE_1]" in masked

    client = _FakeClient(
        "Обратитесь к [PERSON_1:дат], телефон [PHONE_1]. "
        "Если не отвечает — [PERSON_2:им].")
    resp = shield.messages_create(
        client,
        model="test",
        system="Отвечай по базе школы.",
        messages=[{"role": "user",
                   "content": f"Кто классный руководитель? Вот база: {masked}"}],
    )

    text = resp.content[0].text
    assert "Ивановой Марии Петровне" in text
    assert "8 (900) 123-45-67" in text
    assert "Петров Семён Ильич" in text
    assert "[PERSON_" not in text and "[PHONE_" not in text


def test_пд_не_уходит_в_запрос_к_api(shield):
    client = _FakeClient("Ответ.")
    shield.messages_create(
        client,
        model="test",
        system="Справка по школе. Завуч Иванова Мария Петровна.",
        messages=[{"role": "user",
                   "content": "Дай телефон Ивановой Марии, вот он у меня "
                              "записан: 89001234567, верно?"}],
    )
    sent = json.dumps(client.seen, ensure_ascii=False, default=str)
    assert "Иванова" not in sent and "Ивановой" not in sent
    assert "89001234567" not in sent
    assert "[PERSON_1]" in sent


def test_инструкция_про_падежи_добавляется(shield):
    client = _FakeClient("Ответ.")
    shield.messages_create(client, model="test",
                           messages=[{"role": "user", "content": "Привет"}])
    assert CASE_INSTRUCTION in client.seen["system"]


def test_блочный_контент_маскируется(shield):
    client = _FakeClient("Ок.")
    shield.messages_create(
        client, model="test",
        messages=[{"role": "user", "content": [
            {"type": "text", "text": "Записка от Ивановой Марии Петровны."},
            {"type": "image", "source": {"data": "..."}},
        ]}])
    block_text = client.seen["messages"][0]["content"][0]["text"]
    assert "Ивановой" not in block_text
    assert client.seen["messages"][0]["content"][1]["type"] == "image"


def test_восстановление_никогда_не_валит_ответ(shield):
    # даже если структура ответа неожиданная, бот получает ответ, не исключение
    class Weird:
        content = None
    out = shield.restore_response(Weird())
    assert out is not None
