"""Нормализация находок-людей: любое написание приводится к канону.

«Шумилиной Веронике», «Шумилина В.Н.», «В. Н. Шумилина» — это один человек,
и все формы должны сойтись к одному ключу, иначе метки расползутся.

Канон полного имени: «Фамилия Имя [Отчество]» в именительном падеже,
женская фамилия остаётся женской («Шумилина», не «Шумилин»).
Канон записи с инициалами: «Фамилия И.О.».

Правила выбора:
- роль слова (имя / отчество / фамилия) назначается не по порядку слов,
  а по оценкам морфологического разбора: перебираются варианты назначения,
  выигрывает суммарно самый правдоподобный;
- приведение к именительному падежу делается через inflect с сохранением
  рода найденного разбора, а не через normal_form (normal_form у pymorphy
  теряет женский род фамилий и отчеств);
- слово, которое морфология не знает, не «донормализуется» вслепую:
  несклоняемые фамилии («Райку», «Бондаренко») остаются как есть.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

import pymorphy3

# pymorphy3 берём напрямую, не через natasha.MorphVocab: MorphVocab
# подменяет метод inflect своим API с другими именами граммем,
# и склонение через него молча не работает; pymorphy3 вместо pymorphy2 из-за несовместимости pymorphy2 с python 3.11+
_morph: pymorphy3.MorphAnalyzer | None = None


def morph() -> pymorphy3.MorphAnalyzer:
    """Морфология загружается один раз на процесс (правило 5 проекта)."""
    global _morph
    if _morph is None:
        _morph = pymorphy3.MorphAnalyzer()
    return _morph


@dataclass
class PersonKey:
    """Разобранное имя человека."""
    surname: str = ""          # фамилия в им. падеже, род сохранён: «Шумилина»
    first: str = ""            # имя полностью: «Вероника» ("" если только инициалы)
    middle: str = ""           # отчество полностью или ""
    initials: str = ""         # «В.Н.» если исходник был с инициалами
    gender: str = "unknown"    # male / female / unknown
    raw: str = ""              # как встретилось в тексте

    @property
    def canon(self) -> str:
        if self.first:
            parts = [self.surname, self.first, self.middle]
            return " ".join(p for p in parts if p)
        if self.initials:
            return f"{self.surname} {self.initials}".strip()
        return self.surname

    @property
    def first_initial(self) -> str:
        if self.first:
            return self.first[0]
        if self.initials:
            return self.initials[0]
        return ""


# «Шумилина В.Н.», «Шумилина В. Н», «В.Н. Шумилина»
_RE_SURN_FIRST = re.compile(
    r"^\s*(?P<surn>[А-ЯЁ][а-яё\-]+)\s+(?P<i1>[А-ЯЁ])\s?\.\s?(?:(?P<i2>[А-ЯЁ])\s?\.?)?\s*$"
)
_RE_INIT_FIRST = re.compile(
    r"^\s*(?P<i1>[А-ЯЁ])\s?\.\s?(?:(?P<i2>[А-ЯЁ])\s?\.\s?)?(?P<surn>[А-ЯЁ][а-яё\-]+)\s*$"
)

_CYR_WORD = re.compile(r"[А-ЯЁа-яё][А-ЯЁа-яё\-]+")

_GENDER_GRAM = {"male": "masc", "female": "femn"}


@lru_cache(maxsize=8192)
def _parses(word: str):
    return tuple(morph().parse(word.lower()))


def _score(word: str, grammeme: str) -> float:
    """Насколько слово похоже на имя/фамилию/отчество: лучший score разбора."""
    best = 0.0
    for p in _parses(word):
        if grammeme in p.tag:
            best = max(best, p.score)
    return best


def _best_parse(word: str, grammeme: str, gender: str = "unknown"):
    """Лучший разбор с граммемой; при известном поле — сначала совпадающие."""
    cands = [p for p in _parses(word) if grammeme in p.tag]
    if not cands:
        return None
    gram = _GENDER_GRAM.get(gender)
    if gram:
        matching = [p for p in cands if gram in p.tag]
        if matching:
            cands = matching
    return max(cands, key=lambda p: p.score)


def _to_nominative(word: str, grammeme: str, gender: str = "unknown") -> str | None:
    """Слово -> именительный падеж с сохранением рода.

    normal_form не годится: он превращает «Шумилиной» в «шумилин».
    Род передаётся в inflect явно: без этого pymorphy при переходе
    к именительному тоже уходит в мужскую форму.
    """
    p = _best_parse(word, grammeme, gender)
    if p is None:
        return None
    if "nomn" in p.tag:
        return p.word.capitalize()
    target = {"nomn", "sing"}
    gram = _GENDER_GRAM.get(gender)
    if gram is None:
        # род не подсказан снаружи — берём род самого разбора
        if "femn" in p.tag:
            gram = "femn"
        elif "masc" in p.tag:
            gram = "masc"
    if gram:
        target.add(gram)
    try:
        f = p.inflect(target)
    except Exception:
        f = None
    if f is None:
        try:
            f = p.inflect({"nomn", "sing"})
        except Exception:
            f = None
    return f.word.capitalize() if f else p.normal_form.capitalize()


def _parse_gender(word: str, grammeme: str) -> str:
    p = _best_parse(word, grammeme)
    if p is None:
        return "unknown"
    if "femn" in p.tag:
        return "female"
    if "masc" in p.tag:
        return "male"
    return "unknown"


def normalize_person(raw: str) -> PersonKey:
    """Любое написание человека -> PersonKey с каноном."""
    text = re.sub(r"\s+", " ", raw).strip(" ,;:.()|\\")

    # формы с инициалами разбираются отдельной веткой
    m = _RE_SURN_FIRST.match(text) or _RE_INIT_FIRST.match(text)
    if m:
        surn_raw = m.group("surn")
        initials = m.group("i1") + "."
        if m.group("i2"):
            initials += m.group("i2") + "."
        # «Фамилия И.О.» чаще всего стоит в именительном; если у слова есть
        # именительный разбор — верим ему (и его роду), иначе склоняем лучший
        nomn = [p for p in _parses(surn_raw)
                if "Surn" in p.tag and "nomn" in p.tag]
        if nomn:
            p = max(nomn, key=lambda x: x.score)
            gender = "female" if "femn" in p.tag else (
                "male" if "masc" in p.tag else "unknown")
            surname = p.word.capitalize()
        else:
            gender = _parse_gender(surn_raw, "Surn")
            surname = (_to_nominative(surn_raw, "Surn", gender)
                       or surn_raw.capitalize())
        return PersonKey(surname=surname, initials=initials,
                         gender=gender, raw=raw)

    words = [w for w in _CYR_WORD.findall(text) if len(w) >= 2]

    # отчество однозначно
    middle_raw = None
    rest = []
    for w in words:
        if middle_raw is None and _score(w, "Patr") > 0:
            middle_raw = w
        else:
            rest.append(w)

    gender = _parse_gender(middle_raw, "Patr") if middle_raw else "unknown"

    # роли остальных слов — перебором назначений по суммарной оценке
    first_raw = surname_raw = None
    scored = [(w, _score(w, "Name"), _score(w, "Surn")) for w in rest]
    named = [s for s in scored if s[1] > 0 or s[2] > 0]
    plain = [s[0] for s in scored if s[1] == 0 and s[2] == 0]

    if len(named) >= 2:
        (w1, n1, s1), (w2, n2, s2) = named[0], named[1]
        # вариант А: w1 имя, w2 фамилия; вариант Б: наоборот
        if n1 + s2 >= n2 + s1:
            first_raw, surname_raw = w1, w2
        else:
            first_raw, surname_raw = w2, w1
    elif len(named) == 1:
        w, n, s = named[0]
        # одиночное слово с разбором фамилии — это фамилия, даже если
        # оценка имени формально выше: у косвенных форм («Шумилиной»)
        # pymorphy выдумывает маловероятный разбор-имя, и без этого
        # правила одиночная фамилия заводила второго человека
        if s > 0 and (n == 0 or len(rest) == 1):
            surname_raw = w
        elif n > s:
            first_raw = w
        else:
            surname_raw = w

    # слова без разбора имени/фамилии:
    # - в фамилию берём неизвестное морфологии слово («Райку» как раёк
    #   в дательном не считается: именительного разбора у слова нет,
    #   а несклоняемая фамилия в позиции ФИО правдоподобнее) или слово
    #   без именительного разбора; известное слово в именительном
    #   («Директор») фамилией не становится;
    # - в имя берём только слово, которое морфология вообще не знает
    #   (редкие имена вроде «Эльдус»)
    def _has_known_nomn(w: str) -> bool:
        return any("nomn" in p.tag and p.score >= 0.3 for p in _parses(w))

    for w in plain:
        unknown = not morph().word_is_known(w.lower())
        if surname_raw is None and (unknown or not _has_known_nomn(w)):
            surname_raw = w
        elif first_raw is None and unknown:
            first_raw = w

    first = middle = surname = ""
    if first_raw:
        if gender == "unknown":
            gender = _parse_gender(first_raw, "Name")
        first = (_to_nominative(first_raw, "Name", gender)
                 or first_raw.capitalize())
    if middle_raw:
        middle = (_to_nominative(middle_raw, "Patr", gender)
                  or middle_raw.capitalize())
    if surname_raw:
        if _score(surname_raw, "Surn") > 0:
            if gender == "unknown":
                # род не известен из имени и отчества: если у фамилии есть
                # именительный разбор — верим его роду («Каримова» -> женская)
                nomn = [p for p in _parses(surname_raw)
                        if "Surn" in p.tag and "nomn" in p.tag]
                if nomn:
                    p = max(nomn, key=lambda x: x.score)
                    gender = "female" if "femn" in p.tag else (
                        "male" if "masc" in p.tag else "unknown")
                else:
                    gender = _parse_gender(surname_raw, "Surn")
            surname = (_to_nominative(surname_raw, "Surn", gender)
                       or surname_raw.capitalize())
        else:
            surname = surname_raw.capitalize()

    return PersonKey(surname=surname, first=first, middle=middle,
                     gender=gender, raw=raw)


def same_person(a: PersonKey, b: PersonKey) -> bool:
    """Один ли это человек. Совпадение фамилии обязательно, имя сверяется
    по полному совпадению или по первой букве (инициалу).

    Люди без фамилии («Владимир Аркадьевич») сравниваются по имени
    и отчеству: оба поля обязаны совпасть. Запись с фамилией и запись
    без фамилии не склеиваются — тут безопаснее два человека, чем один."""
    if not a.surname and not b.surname:
        return bool(a.first and b.first and a.middle and b.middle
                    and a.first.lower() == b.first.lower()
                    and a.middle.lower() == b.middle.lower())
    if not a.surname or not b.surname:
        return False
    if a.surname.lower() != b.surname.lower():
        return False
    ai, bi = a.first_initial, b.first_initial
    if ai and bi and ai != bi:
        return False
    if a.first and b.first and a.first.lower() != b.first.lower():
        return False
    return True


def merge(base: PersonKey, extra: PersonKey) -> PersonKey:
    """Дополняет запись более полной формой: инициалы + полное имя -> полное."""
    return PersonKey(
        surname=base.surname or extra.surname,
        first=base.first or extra.first,
        middle=base.middle or extra.middle,
        initials=base.initials or extra.initials,
        gender=base.gender if base.gender != "unknown" else extra.gender,
        raw=base.raw,
    )
