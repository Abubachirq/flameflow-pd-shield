"""Masker: находит ПД в тексте и заменяет метками через vault.

Детерминированно: одно значение всегда получает одну метку
в рамках проекта, номера выдаёт vault и хранит их между запусками.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from .config import ShieldConfig
from .detectors import NameDictionary, detect, normalize_phone
from .normalize import normalize_person
from .vault import Vault

log = logging.getLogger("pd_shield")


@dataclass
class Replacement:
    label: str
    type: str
    original: str


class Masker:
    def __init__(self, config: ShieldConfig, vault: Vault):
        self.config = config
        self.vault = vault
        self.dictionary = NameDictionary(config.names)

    def _known_surnames(self) -> set[str]:
        """Фамилии людей, уже записанных в vault: одиночное упоминание
        такой фамилии маскируется, даже если NER в нём не уверен."""
        out = set()
        for entry in self.vault._data["persons"]:
            if entry["surname"]:
                out.add(entry["surname"].lower())
        return out

    def _label_for_span(self, span) -> str | None:
        if span.type == "person":
            key = normalize_person(span.text)
            if not key.surname and not key.first:
                return None  # мусорный спан, человека в нём не разобрать
            return self.vault.label_for_person(key)
        if span.type == "phone":
            return self.vault.label_for_value(
                "phone", normalize_phone(span.text), span.text.strip())
        if span.type == "email":
            return self.vault.label_for_value(
                "email", span.text.strip().lower(), span.text.strip())
        if span.type == "birthdate":
            norm = re.sub(r"[.\-/]", ".", span.text.strip())
            return self.vault.label_for_value("birthdate", norm,
                                              span.text.strip())
        if span.type == "address":
            norm = re.sub(r"\s+", " ", span.text.strip().lower())
            return self.vault.label_for_value("address", norm,
                                              span.text.strip())
        return None

    def mask(self, text: str) -> tuple[str, list[Replacement]]:
        """Текст -> (текст с метками, список замен). Vault сохраняется."""
        if not text:
            return text, []
        spans = detect(
            text,
            dictionary=self.dictionary,
            enabled_types=self.config.enabled_types,
            known_surnames=self._known_surnames(),
        )
        replacements: list[Replacement] = []
        out = []
        pos = 0
        for span in spans:
            label = self._label_for_span(span)
            if label is None:
                continue
            out.append(text[pos:span.start])
            out.append(f"[{label}]")
            pos = span.end
            replacements.append(Replacement(label, span.type, span.text))
        out.append(text[pos:])
        self.vault.save()
        return "".join(out), replacements
