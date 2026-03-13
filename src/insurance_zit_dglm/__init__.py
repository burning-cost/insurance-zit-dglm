"""
insurance-zit-dglm: Zero-Inflated Tweedie Double GLM with CatBoost gradient boosting.

Mathematical foundation:
  - Gu (arXiv:2405.14990): ZIT with dispersion modelling and generalised EM
  - So & Valdez (arXiv:2406.16206 / NAAJ 2025): ZIT boosted trees, CatBoost implementation
"""

from insurance_zit_dglm.model import ZITModel, ZITReport
from insurance_zit_dglm.power import estimate_power
from insurance_zit_dglm.calibration import check_balance, BalanceResult

__all__ = [
    "ZITModel",
    "ZITReport",
    "estimate_power",
    "check_balance",
    "BalanceResult",
]

__version__ = "0.1.0"
