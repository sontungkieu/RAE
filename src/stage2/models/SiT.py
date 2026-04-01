from .lightningDiT import LightningDiT
from .DDT import DiTwDDTHead


class SiT(LightningDiT):
    """Repo-facing SiT alias used by the JAX single-tower path."""


class SiTDH(DiTwDDTHead):
    """Repo-facing SiT objective plus DH architecture alias used by JAX-facing configs."""
