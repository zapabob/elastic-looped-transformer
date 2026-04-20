"""HuggingFace Hub compatibility layer for ELT-LM.

Exports `ELTConfig` and `ELTForCausalLM`, which wrap the internal `ModelConfig`
and `ELTLanguageModel` to match the HF `transformers` API. Use with
`trust_remote_code=True` once pushed to the Hub, e.g.:

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained("zapabob/elt-lm-base-275m")
    model = AutoModelForCausalLM.from_pretrained(
        "zapabob/elt-lm-base-275m", trust_remote_code=True,
    )
"""

from elt_lm.hf.configuration_elt import ELTConfig
from elt_lm.hf.modeling_elt import ELTForCausalLM

__all__ = ["ELTConfig", "ELTForCausalLM"]
