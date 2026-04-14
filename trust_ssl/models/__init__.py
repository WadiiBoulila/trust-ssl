"""Model factory. Dispatches to the requested method."""

from .backbone import build_backbone
from .simclr import SimCLR
from .byol import BYOL
from .vicreg import VICReg
from .trust_ssl import TrustSSL

_METHODS = {
    "simclr": SimCLR,
    "byol": BYOL,
    "vicreg": VICReg,
    "trust_ssl": TrustSSL,
    "trust_ssl_scalar": TrustSSL,   # ablation: num_factors = 1
    "trust_ssl_cosine": TrustSSL,   # ablation: learned cosine gate, no evidence heads
}


def build_model(cfg, method: str):
    """Instantiate a model for the given method name.

    The ablation variants re-use the TrustSSL class with configuration
    overrides so that every code path is exercised identically.
    """
    if method not in _METHODS:
        raise ValueError(f"unknown method '{method}', choose from {sorted(_METHODS)}")

    if method == "trust_ssl_scalar":
        cfg = cfg.copy()
        cfg["num_factors"] = 1
        cfg["gate_kind"] = "evidential"
    elif method == "trust_ssl_cosine":
        cfg = cfg.copy()
        cfg["gate_kind"] = "cosine"
    elif method == "trust_ssl":
        cfg = cfg.copy()
        cfg["gate_kind"] = "evidential"

    return _METHODS[method](cfg)


__all__ = ["build_model", "build_backbone", "SimCLR", "BYOL", "VICReg", "TrustSSL"]
