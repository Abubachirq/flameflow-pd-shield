"""Конфиг проекта: типы ПД, путь к vault, словарь имён клиента.

Настройки живут в json проекта бота, словарь имён — в переменной
окружения: имена это сами ПД, в файл под версионным контролем им
нельзя. В коде пакета клиентского нет (правило 4 проекта).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

ALL_TYPES = ("person", "phone", "email", "birthdate", "address")


@dataclass
class ShieldConfig:
    names: list[str] = field(default_factory=list)
    names_env: str = "PD_NAMES"
    enabled_types: list[str] = field(default_factory=lambda: list(ALL_TYPES))
    vault_path: str = "pd_vault.enc"
    key_env: str = "PD_SHIELD_KEY"

    def __post_init__(self):
        unknown = set(self.enabled_types) - set(ALL_TYPES)
        if unknown:
            raise ValueError(
                f"Неизвестные типы ПД в конфиге: {sorted(unknown)}. "
                f"Допустимые: {list(ALL_TYPES)}")
        if not self.names:
            self.names = self._names_from_env()

    def _names_from_env(self) -> list[str]:
        """Словарь имён из переменной окружения. По образцу key:
        имена это ПД, в файле конфига им не место."""
        raw = os.environ.get(self.names_env, "")
        if not raw:
            return []
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Переменная окружения {self.names_env} содержит не JSON: "
                f'{e}. Ожидается список строк, например ["Иванов"].') from e
        if not isinstance(data, list) or not all(
                isinstance(x, str) for x in data):
            raise ValueError(
                f"Переменная окружения {self.names_env}: ожидается список "
                f"строк, получено {type(data).__name__}.")
        return data

    @classmethod
    def from_json(cls, path: str) -> "ShieldConfig":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        allowed = {"names", "names_env", "enabled_types", "vault_path",
                   "key_env"}
        unknown = set(data) - allowed
        if unknown:
            raise ValueError(
                f"Неизвестные поля в {path}: {sorted(unknown)}. "
                f"Допустимые: {sorted(allowed)}")
        return cls(**data)

    @property
    def key(self) -> bytes:
        """Ключ шифрования vault из переменной окружения. Никогда из файла."""
        value = os.environ.get(self.key_env, "")
        if not value:
            raise RuntimeError(
                f"Переменная окружения {self.key_env} не задана. "
                f"Сгенерировать ключ: python -m pd_shield.vault --new-key")
        return value.encode()
