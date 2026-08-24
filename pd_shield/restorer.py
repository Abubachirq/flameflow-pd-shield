"""Restorer: метки в ответе LLM -> реальные значения.

Склонение: падеж выбирает LLM тегом в метке ([PERSON_1:дат]),
склоняет локально petrovich. ПД наружу не уходят: модель знает
только метку и падеж.

Деградация всегда безопасная (правило 8 проекта):
- метка без тега или с кривым тегом -> именительный падеж;
- выдуманная метка -> «[неизвестно]» плюс запись в лог;
- ошибка склонения -> имя в именительном.
"""

from __future__ import annotations

import logging
import re

from petrovich.enums import Case, Gender
from petrovich.main import Petrovich

from .vault import Vault

log = logging.getLogger("pd_shield")

# [PERSON_1], [PERSON_1:дат], [PHONE_2] и терпимость к пробелам внутри
RE_LABEL = re.compile(
    r"\[\s*(?P<label>(?:PERSON|PHONE|EMAIL|BIRTHDATE|ADDR)_\d+)"
    r"(?:\s*:\s*(?P<case>им|род|дат|вин|тв|пр))?\s*\]"
)

_CASES = {
    "им": None,  # именительный — исходная форма, склонять нечего
    "род": Case.GENITIVE,
    "дат": Case.DATIVE,
    "вин": Case.ACCUSATIVE,
    "тв": Case.INSTRUMENTAL,
    "пр": Case.PREPOSITIONAL,
}

_GENDERS = {
    "male": Gender.MALE,
    "female": Gender.FEMALE,
    "unknown": Gender.ANDRGN,
}

_petrovich = Petrovich()

# для склонения через pymorphy: тег падежа -> граммема
_OC_CASE = {"род": "gent", "дат": "datv", "вин": "accs",
            "тв": "ablt", "пр": "loct"}
_OC_GENDER = {"male": "masc", "female": "femn"}
_KIND_GRAM = {"first": "Name", "last": "Surn", "middle": "Patr"}


def _decline_part(value: str, kind: str, case_tag: str, gender: str) -> str:
    """Одна часть имени в нужный падеж; при ошибке — как было.

    Сначала pymorphy: он лучше знает реальные имена («Ольги», а не
    «Ольгы» у petrovich). Если pymorphy слово не разбирает — petrovich,
    он предсказывает по окончаниям. Последний запас — исходная форма.
    """
    if not value:
        return value
    from .normalize import _parses  # ленивый импорт, без цикла

    case_gram = _OC_CASE[case_tag]
    gram = _KIND_GRAM[kind]
    gender_gram = _OC_GENDER.get(gender)

    cands = [p for p in _parses(value) if gram in p.tag]
    if gender_gram:
        matching = [p for p in cands if gender_gram in p.tag]
        if matching:
            cands = matching
    if cands:
        p = max(cands, key=lambda x: x.score)
        target = {case_gram, "sing"}
        if gender_gram:
            target.add(gender_gram)
        try:
            f = p.inflect(target) or p.inflect({case_gram, "sing"})
        except Exception:
            f = None
        if f:
            return f.word.capitalize()

    try:
        fn = {"first": _petrovich.firstname,
              "last": _petrovich.lastname,
              "middle": _petrovich.middlename}[kind]
        pv_case = _CASES[case_tag]
        pv_gender = _GENDERS.get(gender, Gender.ANDRGN)
        return fn(value, pv_case, pv_gender) or value
    except Exception:
        return value


def _person_text(entry: dict, case_tag: str | None) -> str:
    """Канон человека в нужном падеже: «Фамилия Имя [Отчество]»."""
    surname, first, middle = entry["surname"], entry["first"], entry["middle"]
    initials = entry["initials"]
    tag = case_tag if case_tag in _OC_CASE else None

    if tag is None:  # именительный или тег не распознан
        if first:
            return " ".join(p for p in (surname, first, middle) if p)
        return f"{surname} {initials}".strip() if initials else surname

    gender = entry.get("gender", "unknown")
    s = _decline_part(surname, "last", tag, gender)
    if first:
        f = _decline_part(first, "first", tag, gender)
        m = _decline_part(middle, "middle", tag, gender) if middle else ""
        return " ".join(p for p in (s, f, m) if p)
    # человек известен только как «Фамилия И.О.»: склоняется фамилия,
    # инициалы не меняются
    return f"{s} {initials}".strip() if initials else s


class Restorer:
    def __init__(self, vault: Vault):
        self.vault = vault

    def restore(self, text: str) -> str:
        """Ответ LLM с метками -> текст для пользователя."""
        if not text:
            return text

        out = []
        pos = 0
        for m in RE_LABEL.finditer(text):
            label = m.group("label")
            case_tag = m.group("case")
            found = self.vault.lookup(label)
            if found is None:
                log.warning("pd_shield: LLM выдал неизвестную метку %r", label)
                value = "[неизвестно]"
            else:
                kind, payload = found
                value = (_person_text(payload, case_tag)
                         if kind == "person" else payload)
            # «Сидорова В.А.» перед точкой конца предложения — без двойной точки
            if value.endswith(".") and text[m.end():m.end() + 1] == ".":
                value = value[:-1]
            out.append(text[pos:m.start()])
            out.append(value)
            pos = m.end()
        out.append(text[pos:])
        return "".join(out)
