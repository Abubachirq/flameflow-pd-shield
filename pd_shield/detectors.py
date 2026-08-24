"""Детекторы персональных данных в тексте.

Те же детекторы, что в инвентаризационном pd_scan шага 1
(один источник истины), плюс словарь известных имён из конфига клиента.

Каждый детектор возвращает спаны: (start, end, type, text).
Типы: person, phone, email, birthdate, address.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

from .normalize import _parses, _score, morph, normalize_person  # noqa: F401

# NER natasha загружается лениво и один раз (правило 5 проекта):
# эмбеддинги NER — это сотни мегабайт при повторной загрузке
_ner_bundle = None


def _ner():
    global _ner_bundle
    if _ner_bundle is None:
        from natasha import Segmenter, NewsEmbedding, NewsNERTagger
        seg = Segmenter()
        emb = NewsEmbedding()
        tagger = NewsNERTagger(emb)
        _ner_bundle = (seg, tagger)
    return _ner_bundle


@dataclass(frozen=True)
class Span:
    start: int
    end: int
    type: str      # person / phone / email / birthdate / address
    text: str
    source: str    # ner / dict / initials / case / regex


# ---------------------------------------------------------------- регулярки

RE_EMAIL = re.compile(r"[A-Za-z0-9_.+\-]+@[A-Za-z0-9\-]+\.[A-Za-z0-9\-.]+")

RE_PHONE = re.compile(
    r"(?<!\d)(?:\+7|8|7)[\s\-().]{0,3}\d{3}[\s\-().]{0,3}\d{3}"
    r"[\s\-().]{0,3}\d{2}[\s\-().]{0,3}\d{2}(?!\d)"
    r"|(?<!\d)\+\d{10,15}(?!\d)"
)

RE_DATE = re.compile(r"(?<!\d)([0-3]?\d[./\-][01]?\d[./\-](?:19|20)\d{2})(?!\d)")
BIRTH_CTX = re.compile(r"рожд|д\.\s?р\.|дата\s+рожд|год\s+рожд", re.IGNORECASE)

RE_ADDR = re.compile(
    r"(?:г\.\s?[А-ЯЁ][а-яё\-]+|город\s[А-ЯЁ][а-яё\-]+)?[^\n]{0,40}?"
    r"(?:ул\.|улица|просп\.|проспект|пер\.|переулок|пр-т|бульвар|б-р|шоссе|наб\.|мкр)"
    r"\s?[А-ЯЁ0-9][^\n]{0,50}?(?:д\.|дом)\s?\d+[^\n]{0,25}",
)

RE_INITIALS = re.compile(
    r"\b[А-ЯЁ][а-яё\-]{2,}\s+[А-ЯЁ]\s?\.\s?(?:[А-ЯЁ]\s?\.?)?"
    r"|\b[А-ЯЁ]\s?\.\s?(?:[А-ЯЁ]\s?\.\s?)?[А-ЯЁ][а-яё\-]{2,}\b"
)

_SURN_SUFFIX = (
    r"(?:ов|ев|ёв|ин|ын|ск|цк)"
    r"(?:а|у|ым|ом|е|ой|ую|ая|ий|ого|ому|им|их|ых|ые|ей|ою)?"
)
RE_CASE_FIO = re.compile(
    r"\b[А-ЯЁ][а-яё]{2,}(?:\s+[А-ЯЁ][а-яё]{2,})?\s+[А-ЯЁ][а-яё]*" + _SURN_SUFFIX + r"\b"
    r"|\b[А-ЯЁ][а-яё]*" + _SURN_SUFFIX + r"\s+[А-ЯЁ][а-яё]{2,}(?:\s+[А-ЯЁ][а-яё]{2,})?\b"
)

_CYR_TOKEN = re.compile(r"[А-ЯЁ][а-яё\-]+")

STOP_WORDS = {
    "заказчик", "заказчика", "заказчику", "заказчиком", "исполнитель",
    "исполнителя", "исполнителю", "исполнителем", "директор", "школа",
    "школы", "школе", "положение", "правила", "договор", "инструкция",
    "алгоритм", "куратор", "куратора", "методист", "ученик", "ученика",
    "учитель", "учителя", "родитель", "родителя", "ребёнок", "ребенка",
    "россия", "россии", "федерации",
}

NER_CHUNK = 40_000


def normalize_phone(text: str) -> str:
    digits = re.sub(r"\D", "", text)
    if len(digits) == 11 and digits.startswith("8"):
        digits = "7" + digits[1:]
    return digits


# ---------------------------------------------------------------- словарь

class NameDictionary:
    """Словарь известных имён из конфига клиента.

    Ловит людей по лемме фамилии и имени: любая падежная форма
    слова из словаря считается находкой. Это добор к NER для редких
    фамилий и обязательная страховка: имена, которые клиент назвал
    сам, не должны зависеть от чутья модели.
    """

    def __init__(self, names: Iterable[str]):
        self.lemmas: set[str] = set()
        self.exact: set[str] = set()
        for name in names:
            key = normalize_person(name)
            for part in (key.surname, key.first, key.middle):
                if part:
                    self.exact.add(part.lower())
                    for p in _parses(part):
                        self.lemmas.add(p.normal_form)

    def _word_hits(self, word: str) -> bool:
        lower = word.lower()
        if lower in self.exact:
            return True
        return any(p.normal_form in self.lemmas for p in _parses(word))

    def spans(self, text: str) -> list[Span]:
        if not self.lemmas:
            return []
        out = []
        run: list[tuple[int, int]] = []  # подряд идущие словарные слова
        for m in _CYR_TOKEN.finditer(text):
            if self._word_hits(m.group(0)):
                # Внутри одного имени слова разделяются только пробелами.
                # Любой другой разделитель — запятая, скобка, союз «и» —
                # это граница между людьми. Союз в токены не попадает
                # (нужна заглавная), поэтому смотрим сам разрыв, а не длину:
                # « и » — те же три символа, что и прежний допуск, и
                # проскакивал, склеивая двух человек в одну метку.
                if run and text[run[-1][1]:m.start()].strip():
                    out.append(self._flush(text, run))
                    run = []
                run.append((m.start(), m.end()))
            elif run:
                out.append(self._flush(text, run))
                run = []
        if run:
            out.append(self._flush(text, run))
        return out

    @staticmethod
    def _flush(text: str, run: list[tuple[int, int]]) -> Span:
        s, e = run[0][0], run[-1][1]
        return Span(s, e, "person", text[s:e], "dict")


# ---------------------------------------------------------------- детекторы

def regex_spans(text: str) -> list[Span]:
    out = []
    for m in RE_EMAIL.finditer(text):
        out.append(Span(m.start(), m.end(), "email", m.group(0), "regex"))
    for m in RE_PHONE.finditer(text):
        digits = re.sub(r"\D", "", m.group(0))
        if len(digits) in (10, 11, 12) and len(set(digits)) > 2:
            out.append(Span(m.start(), m.end(), "phone", m.group(0), "regex"))
    for m in RE_DATE.finditer(text):
        ctx = text[max(0, m.start() - 60):m.end() + 60]
        if BIRTH_CTX.search(ctx):
            out.append(Span(m.start(1), m.end(1), "birthdate",
                            m.group(1), "regex"))
    for m in RE_ADDR.finditer(text):
        out.append(Span(m.start(), m.end(), "address", m.group(0), "regex"))
    for m in RE_INITIALS.finditer(text):
        first_word = _CYR_TOKEN.search(m.group(0))
        if first_word and first_word.group(0).lower() in STOP_WORDS:
            continue
        out.append(Span(m.start(), m.end(), "person", m.group(0), "initials"))
    for m in RE_CASE_FIO.finditer(text):
        words = [w.lower() for w in m.group(0).split()]
        if any(w in STOP_WORDS for w in words):
            continue
        out.append(Span(m.start(), m.end(), "person", m.group(0), "case"))
    return out


def ner_spans(text: str, known_surnames: set[str] | None = None,
              include_single: bool = False) -> list[Span]:
    """ФИО через natasha NER.

    Спан из одного слова — только если слово совпадает со словарём
    или с фамилией уже известного человека: одиночные срабатывания
    NER шумят (урок инвентаризации шага 1), а замаскированное лишнее
    слово портит текст для поиска. include_single=True снимает фильтр —
    это режим инвентаризации (pd-scan), где лучше перебрать, чем недобрать.
    """
    from natasha import Doc
    seg, tagger = _ner()
    known = {s.lower() for s in (known_surnames or set())}
    out = []
    for offset in range(0, len(text), NER_CHUNK):
        chunk = text[offset:offset + NER_CHUNK]
        if not re.search(r"[А-ЯЁ]", chunk):
            continue
        doc = Doc(chunk)
        doc.segment(seg)
        doc.tag_ner(tagger)
        for span in doc.spans:
            if span.type != "PER":
                continue
            val = span.text.strip()
            if len(val) < 3 or val.lower() in STOP_WORDS:
                continue
            single = " " not in val
            if single and not include_single:
                lemmas = {p.normal_form for p in _parses(val)}
                if val.lower() not in known and not (lemmas & known):
                    continue
            out.append(Span(offset + span.start, offset + span.stop,
                            "person", val, "ner"))
    return out


def name_patr_spans(text: str) -> list[Span]:
    """Имя с отчеством без фамилии: «Владимир Аркадьевич».

    NER такие пары нестабильно распознаёт (имя может совпадать
    с городом), регулярки требуют фамилию. Морфология надёжнее:
    два слова подряд, первое разбирается как имя, второе как отчество.
    Пропуск найден инвентаризацией боевой базы 02.08.2026.
    """
    out = []
    tokens = list(_CYR_TOKEN.finditer(text))
    for a, b in zip(tokens, tokens[1:]):
        if b.start() - a.end() > 2:
            continue
        w1, w2 = a.group(0), b.group(0)
        if w1.lower() in STOP_WORDS or len(w1) < 3:
            continue
        if _score(w1, "Name") > 0 and _score(w2, "Patr") > 0:
            out.append(Span(a.start(), b.end(), "person",
                            text[a.start():b.end()], "name_patr"))
    return out


_INITIAL_TOKEN = re.compile(r"^[А-ЯЁ]\.?$")


def _is_name_word(word: str, protected: set[str] | None = None) -> bool:
    """Слово может быть частью имени: инициал, слово из словаря клиента,
    неизвестное морфологии слово или слово с разбором имени/фамилии/отчества."""
    if _INITIAL_TOKEN.match(word):
        return True
    clean = word.strip(".,;:()«»\"'")
    if not clean:
        return False
    if _INITIAL_TOKEN.match(clean):
        return True
    if protected and clean.lower() in protected:
        return True
    if not morph().word_is_known(clean.lower()):
        return True
    return any(g in p.tag for p in _parses(clean)
               for g in ("Name", "Surn", "Patr"))


def merge_person_spans(text: str, spans: list[Span],
                       protected: set[str] | None = None) -> list[Span]:
    """Пересекающиеся спаны людей объединяются, потом подрезаются.

    Зачем объединять: регулярка ловит «Ответственный Петров», NER —
    «Петров Семён Ильич»; при выборе одного из двух полное ФИО теряется.
    Объединение даёт «Ответственный Петров Семён Ильич», подрезка
    убирает «Ответственный»: должность — не персональные данные,
    и из текста она пропадать не должна.
    """
    persons = sorted([s for s in spans if s.type == "person"],
                     key=lambda s: s.start)
    rest = [s for s in spans if s.type != "person"]
    merged: list[list[int]] = []
    for s in persons:
        if merged and s.start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], s.end)
        else:
            merged.append([s.start, s.end])

    out = []
    for start, end in merged:
        # подрезка краёв: слова, которые не могут быть частью имени
        words = [(m.start() + start, m.end() + start, m.group(0))
                 for m in re.finditer(r"\S+", text[start:end])]
        while words and not _is_name_word(words[0][2], protected):
            words.pop(0)
        while words and not _is_name_word(words[-1][2], protected):
            words.pop()
        if not words:
            continue
        s, e = words[0][0], words[-1][1]
        out.append(Span(s, e, "person", text[s:e], "merged"))
    return rest + out


def resolve(spans: list[Span], enabled_types: list[str]) -> list[Span]:
    """Пересечения: выигрывает более длинный спан, при равенстве — словарный."""
    prio = {"dict": 0, "ner": 1, "initials": 2, "case": 3, "regex": 4}
    spans = [s for s in spans if s.type in enabled_types]
    spans.sort(key=lambda s: (s.start, -(s.end - s.start), prio.get(s.source, 9)))
    out: list[Span] = []
    for s in spans:
        if out and s.start < out[-1].end:
            continue
        out.append(s)
    return out


def detect(text: str, dictionary: NameDictionary | None = None,
           enabled_types: list[str] | None = None,
           known_surnames: set[str] | None = None) -> list[Span]:
    """Все ПД-спаны текста, без пересечений, слева направо."""
    enabled = enabled_types or ["person", "phone", "email",
                                "birthdate", "address"]
    spans = regex_spans(text)
    if "person" in enabled:
        dict_surnames = set()
        if dictionary is not None:
            spans += dictionary.spans(text)
            dict_surnames = dictionary.exact
        spans += name_patr_spans(text)
        spans += ner_spans(text, known_surnames=(known_surnames or set())
                           | dict_surnames)
        spans = merge_person_spans(
            text, spans,
            protected=dict_surnames | {s.lower() for s in (known_surnames or set())})
    return resolve(spans, enabled)
