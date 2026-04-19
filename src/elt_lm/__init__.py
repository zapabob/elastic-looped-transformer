"""ELT (Elastic Looped Transformer) + ILSD — causal-LM port.

Reference: Goyal et al., "ELT: Elastic Looped Transformers for Visual Generation",
arXiv:2604.09168v2 (2026).
"""

from elt_lm.config import ILSDConfig, ModelConfig, TrainConfig
from elt_lm.model import ELTLanguageModel

__all__ = ["ModelConfig", "TrainConfig", "ILSDConfig", "ELTLanguageModel"]
__version__ = "0.1.0"
