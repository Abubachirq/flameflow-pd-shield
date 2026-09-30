"""Детекторы персональных данных в тексте.

Те же детекторы, что в инвентаризационном pd_scan шага 1
(один источник истины), плюс словарь известных имён из конфига клиента.

Каждый детектор возвращает спаны: (start, end, type, text).
Типы: person, phone, email, birthdate, address.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher
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
    # написание для разбора имени, если оно отличается от текста:
    # «тимофеву» (строчными, с опечаткой) разбирается как «Тимофееву»,
    # чтобы получить метку того же человека, что и правильное написание
    canon: str | None = None


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
# для словаря: слово с любой буквы — в чатах имена пишут строчными
_CYR_TOKEN_ANY = re.compile(r"[А-ЯЁа-яё][а-яё\-]+")

# Опечатка в фамилии из словаря: насколько написанное слово должно быть
# похоже на одну из падежных форм фамилии (1.0 — совпадает целиком).
# 0.8 ловит одну пропущенную, лишнюю или перепутанную букву в фамилии
# от 5 букв; проверяются только слова, которых морфология не знает.
TYPO_SIMILARITY = 0.8
TYPO_MIN_LEN = 5

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

    Слово с заглавной буквы — находка, как и раньше. Слово строчными
    или с опечаткой тоже находка, но осторожнее: имя строчными («роман»,
    «вера») часто обычное слово, поэтому оно маскируется только рядом
    с другим словом из словаря («ольге викторовне»). Фамилия строчными
    маскируется и одна. Опечатка ищется только в фамилиях и только
    в словах, которых морфология не знает: «смирно» не станет Смирновой.
    Фамилия, совпадающая с обычным словом, строчными в своей словарной
    форме маскируется всегда («козлов» при Козлове: имя важнее текста),
    в других формах — только рядом с именем («кузнецов» при Кузнецовой).

    Известные ограничения: опечатка, превратившая фамилию в настоящее
    слово языка, не ловится; незнакомое морфологии слово, похожее на
    фамилию из словаря («Тимофеевка» при Тимофееве), маскируется;
    имя с отчеством без фамилии получает свою метку, не метку человека
    с фамилией (так было и для заглавных букв).
    """

    def __init__(self, names: Iterable[str]):
        self.lemmas: set[str] = set()
        self.exact: set[str] = set()
        self.surname_lemmas: set[str] = set()
        self.surname_exact: set[str] = set()
        # падежная форма фамилии строчными -> та же форма с заглавной
        self.surname_forms: dict[str, str] = {}
        self._typo_cache: dict[str, str | None] = {}
        for name in names:
            key = normalize_person(name)
            for part in (key.surname, key.first, key.middle):
                if part:
                    self.exact.add(part.lower())
                    for p in _parses(part):
                        self.lemmas.add(p.normal_form)
            if key.surname:
                surname = key.surname.lower()
                self.surname_exact.add(surname)
                self.surname_forms[surname] = key.surname
                for p in _parses(surname):
                    if "Surn" not in p.tag:
                        continue
                    self.surname_lemmas.add(p.normal_form)
                    for form in p.lexeme:
                        self.surname_forms.setdefault(
                            form.word, form.word.capitalize())

    def _word_hits(self, word: str) -> bool:
        lower = word.lower()
        if lower in self.exact:
            return True
        return any(p.normal_form in self.lemmas for p in _parses(word))

    def _surname_hits(self, word: str) -> bool:
        if word.lower() in self.surname_exact:
            return True
        return any("Surn" in p.tag and p.normal_form in self.surname_lemmas
                   for p in _parses(word))

    @staticmethod
    def _is_common_word(word: str) -> bool:
        """У слова есть заметный разбор обычным словом, не именем:
        «козлов» — это и фамилия, и родительный падеж от «козлы»."""
        lower = word.lower()
        if not morph().word_is_known(lower):
            return False
        return any(p.score >= 0.1 and not any(g in p.tag for g in ("Name", "Surn", "Patr"))
                   for p in _parses(lower))

    def _typo_of_surname(self, word: str) -> str | None:
        """Форма фамилии из словаря, опечаткой которой похоже слово, или None.
        Кеш: при индексации документов одно слово встречается много раз."""
        key = word.lower()
        if key not in self._typo_cache:
            self._typo_cache[key] = self._find_typo(key)
        return self._typo_cache[key]

    def _find_typo(self, word: str) -> str | None:
        lower = word.replace("ё", "е")
        if len(lower) < TYPO_MIN_LEN or morph().word_is_known(lower):
            return None
        best, best_ratio = None, TYPO_SIMILARITY
        for form, display in self.surname_forms.items():
            form_e = form.replace("ё", "е")
            if form_e[0] != lower[0] or abs(len(form_e) - len(lower)) > 2:
                continue
            ratio = SequenceMatcher(None, lower, form_e).ratio()
            if ratio >= best_ratio:
                best, best_ratio = display, ratio
        return best

    def _classify(self, word: str) -> tuple[str | None, str | None]:
        """(сила находки, написание для разбора).

        strong — маскируется и одно; weak — только рядом с другой находкой;
        None — не имя из словаря.
        """
        if word[0].isupper() and self._word_hits(word):
            return "strong", None
        if word[0].islower() and self._word_hits(word):
            # редкую фамилию морфология не разбирает как фамилию,
            # тогда её выдаёт сходство с падежной формой из словаря
            typo = self._typo_of_surname(word)
            # фамилия ровно как в списке маскируется всегда: утечка имени
            # хуже, чем метка на месте совпавшего с ней обычного слова
            exact = word.lower() in self.surname_exact
            surname = exact or ((self._surname_hits(word) or typo)
                                and not self._is_common_word(word))
            kind = "strong" if surname else "weak"
            return kind, typo or word.capitalize()
        typo = self._typo_of_surname(word)
        if typo:
            return "strong", typo
        return None, None

    def spans(self, text: str) -> list[Span]:
        if not self.lemmas:
            return []
        out = []
        # подряд идущие словарные слова: (начало, конец, сила, написание)
        run: list[tuple[int, int, str, str | None]] = []
        for m in _CYR_TOKEN_ANY.finditer(text):
            kind, canon = self._classify(m.group(0))
            if kind:
                # Внутри одного имени слова разделяются только пробелами.
                # Любой другой разделитель — запятая, скобка — это граница
                # между людьми, поэтому смотрим сам разрыв, а не его длину.
                # Союз «и» — тоже слово: не будучи именем из словаря, он
                # прерывает пробег (ветка ниже), и двое не склеиваются.
                if run and text[run[-1][1]:m.start()].strip():
                    out.append(self._flush(text, run))
                    run = []
                run.append((m.start(), m.end(), kind, canon))
            elif run:
                out.append(self._flush(text, run))
                run = []
        if run:
            out.append(self._flush(text, run))
        return [s for s in out if s is not None]

    @staticmethod
    def _flush(text: str, run: list[tuple[int, int, str, str | None]]) -> Span | None:
        kinds = [r[2] for r in run]
        if "strong" not in kinds and len(kinds) < 2:
            return None  # одно имя строчными — скорее обычное слово
        s, e = run[0][0], run[-1][1]
        canon = None
        if any(r[3] for r in run):
            canon = text[s:e]
            for start, end, _, word in reversed(run):
                if word:
                    canon = canon[:start - s] + word + canon[end - s:]
        return Span(s, e, "person", text[s:e], "dict", canon)


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
        # исправленное написание словарных находок переносится в итог
        canon_spans = sorted((p for p in persons if p.canon
                              and p.start >= s and p.end <= e),
                             key=lambda p: p.start, reverse=True)
        canon = None
        if canon_spans:
            canon = text[s:e]
            for p in canon_spans:
                canon = canon[:p.start - s] + p.canon + canon[p.end - s:]
        out.append(Span(s, e, "person", text[s:e], "merged", canon))
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
