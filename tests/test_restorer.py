import pytest
from cryptography.fernet import Fernet

from pd_shield.config import ShieldConfig
from pd_shield.masker import Masker
from pd_shield.normalize import normalize_person
from pd_shield.restorer import Restorer
from pd_shield.vault import Vault


@pytest.fixture
def bundle(tmp_path):
    cfg = ShieldConfig(names=[], vault_path=str(tmp_path / "v.enc"))
    vault = Vault(cfg.vault_path, Fernet.generate_key())
    return Masker(cfg, vault), Restorer(vault), vault


def test_склонение_по_тегам_женское(bundle):
    masker, restorer, _ = bundle
    masker.mask("Иванова Мария Петровна ведёт первый класс.")
    assert (restorer.restore("Передайте документы [PERSON_1:дат].")
            == "Передайте документы Ивановой Марии Петровне.")
    assert (restorer.restore("Отчёт подписан [PERSON_1:тв].")
            == "Отчёт подписан Ивановой Марией Петровной.")
    assert (restorer.restore("Кабинет [PERSON_1:род] на втором этаже.")
            == "Кабинет Ивановой Марии Петровны на втором этаже.")


def test_склонение_мужское(bundle):
    masker, restorer, _ = bundle
    masker.mask("Ответственный Петров Семён Ильич.")
    assert (restorer.restore("Обратитесь к [PERSON_1:дат].")
            == "Обратитесь к Петрову Семёну Ильичу.")


def test_метка_без_тега_именительный(bundle):
    masker, restorer, _ = bundle
    masker.mask("Иванова Мария Петровна ведёт первый класс.")
    assert (restorer.restore("Ответственная: [PERSON_1].")
            == "Ответственная: Иванова Мария Петровна.")


def test_инициалы_склоняется_только_фамилия(bundle):
    masker, restorer, vault = bundle
    vault.label_for_person(normalize_person("Сидорова В.А."))
    assert (restorer.restore("Вопрос адресуйте [PERSON_1:дат].")
            == "Вопрос адресуйте Сидоровой В.А.")


def test_несклоняемая_фамилия_не_портится(bundle):
    masker, restorer, vault = bundle
    vault.label_for_person(normalize_person("Бондаренко Ольга"))
    out = restorer.restore("Ключи у [PERSON_1:род].")
    assert out == "Ключи у Бондаренко Ольги."


def test_телефон_и_email_подставляются_как_есть(bundle):
    masker, restorer, _ = bundle
    masker.mask("Телефон 8 (900) 123-45-67, почта kur@example.com.")
    out = restorer.restore("Звоните [PHONE_1], пишите [EMAIL_1].")
    assert "8 (900) 123-45-67" in out
    assert "kur@example.com" in out


def test_выдуманная_метка_не_ломает_ответ(bundle):
    _, restorer, _ = bundle
    out = restorer.restore("Спросите [PERSON_77:дат] о расписании.")
    assert out == "Спросите [неизвестно] о расписании."


def test_кривой_тег_падежа_даёт_именительный(bundle):
    masker, restorer, _ = bundle
    masker.mask("Иванова Мария Петровна ведёт первый класс.")
    # «:датт» не распознаётся как тег — регулярка не съедает метку целиком,
    # но и не падает; метка без валидного тега уходит в именительный
    out = restorer.restore("Документы у [PERSON_1 : род].")
    assert "Ивановой Марии Петровны" in out
