"""Vault: зашифрованная таблица метка-значение.

Внутри шифра обычный json:
{
  "counters": {"PERSON": 2, "PHONE": 1},
  "persons": [
    {"label": "PERSON_1", "surname": "Шумилина", "first": "Вероника",
     "middle": "", "initials": "В.Н.", "gender": "female",
     "forms": ["Шумилиной В.Н", "Шумилина Вероника"]}
  ],
  "values": {"phone": {"79171655703": {"label": "PHONE_1",
                                        "display": "+7 917 165 5703"}}}
}

Шифрование Fernet (симметричный шифр из библиотеки cryptography:
один секретный ключ и запирает файл, и отпирает; без ключа файл
выглядит случайными байтами). Ключ приходит из переменной окружения,
в файле и в git его нет.

Запись атомарная: сначала временный файл рядом, потом переименование.
Обрыв процесса не оставляет vault полуписаным.

Файл vault лежит отдельно от базы бота (на Railway — на Volume)
и в саму базу или индекс не попадает никогда.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile

from cryptography.fernet import Fernet, InvalidToken

from .normalize import PersonKey, same_person, merge

log = logging.getLogger("pd_shield")

TYPE_LABEL = {
    "person": "PERSON",
    "phone": "PHONE",
    "email": "EMAIL",
    "birthdate": "BIRTHDATE",
    "address": "ADDR",
}


class VaultError(Exception):
    pass


class Vault:
    def __init__(self, path: str, key: bytes):
        self.path = path
        try:
            self._fernet = Fernet(key)
        except Exception as e:
            raise VaultError(
                "Ключ шифрования не подходит по формату. Ключ Fernet — "
                "это строка из «python -m pd_shield.vault --new-key», "
                f"а не произвольный пароль. Детали: {e}") from e
        self._data = {"counters": {}, "persons": [], "values": {}}
        self._dirty = False
        self._mtime = None
        if os.path.exists(path):
            self._load()

    # ------------------------------------------------------------- хранение

    def _load(self):
        with open(self.path, "rb") as f:
            blob = f.read()
        try:
            raw = self._fernet.decrypt(blob)
        except InvalidToken as e:
            raise VaultError(
                f"Не удалось расшифровать {self.path}: ключ не тот "
                f"или файл повреждён.") from e
        self._data = json.loads(raw.decode("utf-8"))
        try:
            self._mtime = os.path.getmtime(self.path)
        except OSError:
            self._mtime = None

    def _maybe_reload(self):
        """Перечитывает файл, если его изменил кто-то другой.

        В боте живут два экземпляра Vault: у индексации и у ответов.
        Индексация пишет новые метки в файл, а экземпляр answerer
        загрузился при старте процесса и без перечитывания держал бы
        устаревшую копию: restorer не находил бы свежие метки
        (симптом «[неизвестно]» вместо имён, боевой проект, 02.08.2026).
        Несохранённые собственные правки при этом не выбрасываются.
        """
        if self._dirty:
            return
        try:
            mtime = os.path.getmtime(self.path)
        except OSError:
            return
        if mtime != self._mtime:
            self._load()

    def save(self):
        if not self._dirty:
            return
        blob = self._fernet.encrypt(
            json.dumps(self._data, ensure_ascii=False).encode("utf-8"))
        d = os.path.dirname(os.path.abspath(self.path))
        os.makedirs(d, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=d, prefix=".pd_vault_")
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(blob)
            os.replace(tmp, self.path)  # атомарная замена
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)
        self._dirty = False
        try:
            self._mtime = os.path.getmtime(self.path)
        except OSError:
            self._mtime = None

    # ------------------------------------------------------------- метки

    def _next_label(self, type_: str) -> str:
        prefix = TYPE_LABEL[type_]
        n = self._data["counters"].get(prefix, 0) + 1
        self._data["counters"][prefix] = n
        self._dirty = True
        return f"{prefix}_{n}"

    def label_for_person(self, key: PersonKey) -> str:
        """Метка человека. Одна на все формы написания.

        Совпадение ищется через same_person (фамилия + имя/инициал).
        Более полная форма дополняет запись: пришли инициалы, потом
        полное имя — запись обновляется, метка остаётся.
        Неоднозначность (подходят двое) не угадывается: заводится
        отдельная запись, конфликт пишется в лог.
        """
        self._maybe_reload()
        matches = []
        for entry in self._data["persons"]:
            stored = PersonKey(
                surname=entry["surname"], first=entry["first"],
                middle=entry["middle"], initials=entry["initials"],
                gender=entry["gender"])
            if same_person(stored, key):
                matches.append((entry, stored))

        if len(matches) > 1:
            exact = [(e, s) for e, s in matches
                     if s.first and key.first
                     and s.first.lower() == key.first.lower()]
            if len(exact) == 1:
                matches = exact
            else:
                log.warning(
                    "pd_shield: неоднозначное совпадение для %r "
                    "(кандидатов: %d) — завожу отдельную метку",
                    key.canon, len(matches))
                matches = []

        if matches:
            entry, stored = matches[0]
            merged = merge(stored, key)
            entry.update(surname=merged.surname, first=merged.first,
                         middle=merged.middle, initials=merged.initials,
                         gender=merged.gender)
            if key.raw and key.raw not in entry["forms"]:
                entry["forms"].append(key.raw)
            self._dirty = True
            return entry["label"]

        label = self._next_label("person")
        self._data["persons"].append({
            "label": label,
            "surname": key.surname, "first": key.first,
            "middle": key.middle, "initials": key.initials,
            "gender": key.gender,
            "forms": [key.raw] if key.raw else [],
        })
        self._dirty = True
        return label

    def label_for_value(self, type_: str, norm: str, display: str) -> str:
        """Метка для телефона/email/даты/адреса по нормализованному ключу."""
        self._maybe_reload()
        bucket = self._data["values"].setdefault(type_, {})
        if norm in bucket:
            return bucket[norm]["label"]
        label = self._next_label(type_)
        bucket[norm] = {"label": label, "display": display}
        self._dirty = True
        return label

    # ------------------------------------------------------------- обратно

    def lookup(self, label: str):
        """Запись по метке: ('person', entry) или (тип, display) или None."""
        self._maybe_reload()
        for entry in self._data["persons"]:
            if entry["label"] == label:
                return ("person", entry)
        for type_, bucket in self._data["values"].items():
            for rec in bucket.values():
                if rec["label"] == label:
                    return (type_, rec["display"])
        return None


def _new_key():
    print(Fernet.generate_key().decode())


if __name__ == "__main__":
    import sys
    if "--new-key" in sys.argv:
        _new_key()
    else:
        print("Использование: python -m pd_shield.vault --new-key")
