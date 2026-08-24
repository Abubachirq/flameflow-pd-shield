"""flameflow-pd-shield: обезличивание персональных данных для ботов.

Публичный вход:

    from pd_shield import Shield, ShieldConfig

    shield = Shield.from_config_file("pd_config.json")
    text = shield.mask_for_index(text)                  # до индексации
    resp = shield.messages_create(client, **kwargs)     # вместо messages.create
"""

from .config import ShieldConfig
from .integrate import Shield, CASE_INSTRUCTION
from .masker import Masker
from .restorer import Restorer
from .vault import Vault, VaultError

__all__ = ["Shield", "ShieldConfig", "Masker", "Restorer", "Vault",
           "VaultError", "CASE_INSTRUCTION"]
