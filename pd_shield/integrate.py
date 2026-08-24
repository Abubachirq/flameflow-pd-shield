"""Точки подключения в бот: пайплайн загрузки данных и обёртка Claude API.

Подключение в существующий бот — две правки:

1. В пайплайне индексации (между извлечением текста и нарезкой на чанки):

       text = shield.mask_for_index(text)

2. Вместо прямого вызова client.messages.create:

       response = shield.messages_create(client, model=..., messages=[...])

Всё остальное в боте не меняется.
"""

from __future__ import annotations

import logging

from .config import ShieldConfig
from .masker import Masker
from .restorer import Restorer
from .vault import Vault

log = logging.getLogger("pd_shield")

# Инструкция про падежные теги. Добавляется в system каждого вызова:
# LLM знает грамматику предложения, которое пишет, и помечает падеж сам;
# имена ему для этого не нужны.
CASE_INSTRUCTION = (
    "В тексте встречаются метки вида [PERSON_1], [PHONE_2] — это "
    "обезличенные данные, их нельзя раскрывать, выдумывать или изменять. "
    "Используя метку человека в своём ответе, добавь после двоеточия "
    "падеж, которого требует грамматика ТВОЕЙ фразы (не фразы из "
    "документа): им, род, дат, вин, тв или пр. Проверь себя: подставь "
    "мысленно любое имя и просклоняй. Примеры: "
    "«Обратитесь к [PERSON_1:дат]» (к кому), "
    "«Это записка [PERSON_2:род]» (кого), "
    "«Аудит проводится [PERSON_3:тв]» (кем), "
    "«Спросите про [PERSON_4:вин]» (про кого), "
    "«Ответственная — [PERSON_5:им]» (кто). "
    "Метки телефонов, адресов и дат используй без падежа, как есть."
)


class Shield:
    """Единая точка входа: конфиг + vault + masker + restorer."""

    def __init__(self, config: ShieldConfig, key: bytes | None = None):
        self.config = config
        self.vault = Vault(config.vault_path, key or config.key)
        self.masker = Masker(config, self.vault)
        self.restorer = Restorer(self.vault)

    @classmethod
    def from_config_file(cls, path: str) -> "Shield":
        return cls(ShieldConfig.from_json(path))

    # -------------------------------------------------- точка 1: индексация

    def mask_for_index(self, text: str) -> str:
        """Обезличить текст до нарезки на чанки и индексации."""
        masked, _ = self.masker.mask(text)
        return masked

    # -------------------------------------------------- точка 2: Claude API

    def mask_request(self, kwargs: dict) -> dict:
        """Маскирует все текстовые части запроса к Claude API.

        Вопрос пользователя тоже может содержать ПД, поэтому маскируется
        весь запрос, а не только чанки из базы.
        """
        out = dict(kwargs)

        system = out.get("system")
        if isinstance(system, str):
            out["system"] = (self.masker.mask(system)[0]
                             + "\n\n" + CASE_INSTRUCTION)
        else:
            out["system"] = CASE_INSTRUCTION

        messages = []
        for msg in out.get("messages", []):
            msg = dict(msg)
            content = msg.get("content")
            if isinstance(content, str):
                msg["content"] = self.masker.mask(content)[0]
            elif isinstance(content, list):
                blocks = []
                for block in content:
                    block = dict(block)
                    if block.get("type") == "text" and "text" in block:
                        block["text"] = self.masker.mask(block["text"])[0]
                    blocks.append(block)
                msg["content"] = blocks
            messages.append(msg)
        out["messages"] = messages
        return out

    def restore_response(self, response):
        """Восстанавливает реальные значения в текстовых блоках ответа."""
        try:
            for block in response.content:
                if getattr(block, "type", None) == "text":
                    block.text = self.restorer.restore(block.text)
        except Exception:
            log.exception("pd_shield: не удалось восстановить ответ, "
                          "возвращаю как есть (с метками)")
        return response

    def messages_create(self, client, **kwargs):
        """Обёртка client.messages.create: маскировать -> вызвать -> восстановить."""
        masked_kwargs = self.mask_request(kwargs)
        response = client.messages.create(**masked_kwargs)
        return self.restore_response(response)
