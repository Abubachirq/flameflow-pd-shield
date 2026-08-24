import pytest
from cryptography.fernet import Fernet

from pd_shield.config import ShieldConfig
from pd_shield.masker import Masker
from pd_shield.vault import Vault


@pytest.fixture
def masker(tmp_path):
    cfg = ShieldConfig(
        names=["Иванова Мария Петровна", "Райку Валентина"],
        vault_path=str(tmp_path / "v.enc"),
    )
    vault = Vault(cfg.vault_path, Fernet.generate_key())
    return Masker(cfg, vault)


def test_полное_фио_маскируется(masker):
    out, reps = masker.mask("Ответственная за кабинет Иванова Мария Петровна.")
    assert "Иванова" not in out
    assert "[PERSON_1]" in out
    assert reps[0].type == "person"


def test_падежные_формы_дают_ту_же_метку(masker):
    a, _ = masker.mask("Приказ подписан Ивановой Марией Петровной.")
    b, _ = masker.mask("Обратитесь к Ивановой М.П. лично.")
    c, _ = masker.mask("Иванова Мария Петровна ведёт первый класс.")
    assert "[PERSON_1]" in a and "[PERSON_1]" in b and "[PERSON_1]" in c


def test_разные_люди_разные_метки(masker):
    out, _ = masker.mask("Смену принимают Иванова Мария и Петров Семён.")
    assert "[PERSON_1]" in out and "[PERSON_2]" in out


def test_телефоны_в_разных_форматах_одна_метка(masker):
    out, _ = masker.mask(
        "Звонить: 8 (900) 123-45-67, дублирующий +7 900 123 45 67, "
        "запасной 89001234568.")
    assert out.count("[PHONE_1]") == 2  # два формата одного номера
    assert "[PHONE_2]" in out           # другой номер — другая метка
    assert "123-45-67" not in out


def test_email_и_дата_рождения(masker):
    out, _ = masker.mask(
        "Куратор (дата рождения 03.05.1987) пишет с адреса kur@example.com.")
    assert "[EMAIL_1]" in out and "kur@example.com" not in out
    assert "[BIRTHDATE_1]" in out and "03.05.1987" not in out


def test_дата_без_контекста_рождения_не_маскируется(masker):
    out, _ = masker.mask("Собрание перенесли на 03.05.2026, кабинет тот же.")
    assert "03.05.2026" in out


def test_словарь_ловит_несклоняемую_фамилию(masker):
    # NER и регулярки «Райку» в косвенном контексте пропускают,
    # словарь из конфига — обязан поймать
    out, _ = masker.mask("Ключи передайте Райку.")
    assert "Райку" not in out
    assert "[PERSON_" in out


def test_одиночное_слово_без_словаря_не_маскируется(masker):
    out, _ = masker.mask("Заказчик обязан оплатить Диагностику вовремя.")
    assert "Заказчик" in out and "Диагностику" in out


def test_одиночная_фамилия_известного_человека_маскируется(masker):
    masker.mask("Иванова Мария Петровна ведёт первый класс.")
    out, _ = masker.mask("Кабинет закрывает Иванова.")
    assert "Иванова" not in out
    assert "[PERSON_1]" in out


def test_детерминированность_между_экземплярами(tmp_path):
    cfg = ShieldConfig(names=[], vault_path=str(tmp_path / "v.enc"))
    key = Fernet.generate_key()
    m1 = Masker(cfg, Vault(cfg.vault_path, key))
    a, _ = m1.mask("Заявление от Петрова Семёна Ильича, тел. 89001234567.")
    m2 = Masker(cfg, Vault(cfg.vault_path, key))
    b, _ = m2.mask("Петров Семён Ильич, телефон 8 900 123 45 67.")
    assert "[PERSON_1]" in a and "[PERSON_1]" in b
    assert "[PHONE_1]" in a and "[PHONE_1]" in b


def test_отключённые_типы_не_маскируются(tmp_path):
    cfg = ShieldConfig(names=[], enabled_types=["phone"],
                       vault_path=str(tmp_path / "v.enc"))
    m = Masker(cfg, Vault(cfg.vault_path, Fernet.generate_key()))
    out, _ = m.mask("Петров Семён Ильич, телефон 89001234567.")
    assert "Петров" in out
    assert "[PHONE_1]" in out


def test_одиночная_фамилия_в_косвенном_падеже_та_же_метка(tmp_path):
    """Баг с боевого проекта 02.08.2026: «Телефон Ивановой для связи» заводил
    второго человека, потому что одиночная косвенная форма фамилии
    разбиралась как имя."""
    from cryptography.fernet import Fernet
    cfg = ShieldConfig(names=[], vault_path=str(tmp_path / "v.enc"))
    m = Masker(cfg, Vault(cfg.vault_path, Fernet.generate_key()))
    doc = ("Диагностику проводит Иванова Мария Петровна. "
           "Телефон Ивановой для связи: 89001234567.")
    out, reps = m.mask(doc)
    person_labels = {r.label for r in reps if r.type == "person"}
    assert person_labels == {"PERSON_1"}
    assert "Телефон [PERSON_1] для связи" in out


def test_имя_с_отчеством_без_фамилии_маскируется(tmp_path):
    """Пропуск, найденный инвентаризацией 02.08.2026:
    имя с отчеством без фамилии."""
    from cryptography.fernet import Fernet
    cfg = ShieldConfig(names=[], vault_path=str(tmp_path / "v.enc"))
    m = Masker(cfg, Vault(cfg.vault_path, Fernet.generate_key()))
    out, _ = m.mask("Кружок робототехники ведёт Владимир Аркадьевич по средам.")
    assert "Владимир" not in out
    assert "[PERSON_1]" in out
    out2, _ = m.mask("Занятие у Владимира Аркадьевича перенесли.")
    assert "Владимира" not in out2 and "[PERSON_1]" in out2


# --- Границы между людьми: перечисления не должны склеиваться ---
# До 0.1.5 словарный детектор рвал пробег только по длине зазора (>3),
# а « и » — ровно три символа: двое людей получали одну метку, второе имя
# в хранилище не попадало вовсе.

def _метки(text: str) -> set[str]:
    import re
    return set(re.findall(r"PERSON_\d+", text))


def test_двое_через_союз_получают_разные_метки(masker):
    out, _ = masker.mask(
        "Присутствовали Иванова Мария Петровна и Райку Валентина.")
    assert len(_метки(out)) == 2, out


def test_двое_через_запятую_получают_разные_метки(masker):
    out, _ = masker.mask(
        "Присутствовали Иванова Мария Петровна, Райку Валентина.")
    assert len(_метки(out)) == 2, out


def test_одиночные_фамилии_через_союз_не_склеиваются(masker):
    out, _ = masker.mask("Обратитесь к Ивановой и Райку.")
    assert len(_метки(out)) == 2, out


def test_полное_фио_остаётся_одной_меткой(masker):
    """Регресс-страховка: пробелы внутри имени пробег не рвут."""
    out, _ = masker.mask("Ответственная Иванова Мария Петровна.")
    assert len(_метки(out)) == 1, out
    assert "Иванова" not in out and "Петровна" not in out


def test_фамилия_через_дефис_остаётся_одной_меткой(tmp_path):
    """Регресс-страховка: дефис живёт внутри токена, зазора не создаёт."""
    from cryptography.fernet import Fernet
    from pd_shield.config import ShieldConfig
    from pd_shield.masker import Masker
    from pd_shield.vault import Vault
    cfg = ShieldConfig(names=["Иванова-Петрова Мария Сергеевна"],
                       vault_path=str(tmp_path / "d.enc"))
    m = Masker(cfg, Vault(cfg.vault_path, Fernet.generate_key()))
    out, _ = m.mask("Ответственная Иванова-Петрова Мария Сергеевна.")
    assert len(_метки(out)) == 1, out
    assert "Иванова" not in out
