"""Configuration dataclasses for ELT-LM.

Mirrors arXiv:2604.09168 §4 notation:
    N       = number of unique Transformer layers forming composite block g_Theta
    L       = number of times g_Theta is iterated (F_{N,L}(x) = g_Theta^L(x))
    L_min, L_max  = bounds of stochastic student sampling S^3 (L_int ~ U[L_min, L_max])
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import yaml


@dataclass
class ModelConfig:
    vocab_size: int = 248320           # Qwen3.5 tokenizer
    d_model: int = 768
    n_unique_layers: int = 12          # N in paper
    n_heads: int = 12
    n_kv_heads: int | None = None      # None => MHA (== n_heads); set smaller for GQA
    head_dim: int | None = None        # None => d_model // n_heads
    d_ff: int = 2048                   # SwiGLU intermediate
    max_seq_len: int = 2048
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-6
    tie_word_embeddings: bool = True
    dropout: float = 0.0

    L_min: int = 1                     # S^3 lower bound
    L_max: int = 4                     # S^3 upper bound (teacher loop count)

    init_std: float = 0.02
    grad_checkpoint: bool = True       # checkpoint each g_Theta call

    def __post_init__(self) -> None:
        if self.head_dim is None:
            assert self.d_model % self.n_heads == 0, \
                f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
            self.head_dim = self.d_model // self.n_heads
        if self.n_kv_heads is None:
            self.n_kv_heads = self.n_heads
        assert self.n_heads % self.n_kv_heads == 0, \
            "n_heads must be divisible by n_kv_heads for GQA"
        assert 1 <= self.L_min <= self.L_max, "Require 1 <= L_min <= L_max"


@dataclass
class ILSDConfig:
    """Intra-Loop Self-Distillation (paper eq. 3) configuration."""
    enabled: bool = True
    lambda_init: float = 1.0
    lambda_final: float = 0.0
    lambda_anneal_steps: int = 20_000
    # Warmup before enabling S^3 (teacher-only until this step). 0 = start ILSD immediately.
    warmup_steps: int = 2_000
    # If True, sample L_int strictly < L_max so that distillation term is meaningful.
    # If False, L_int == L_max collapses dist term to zero and that's ok (paper wording
    # supports U[L_min, L_max] inclusive).
    strict_student_below_teacher: bool = True
    # Teacher-only temperature + tiny uniform smoothing for softer distillation targets.
    distill_teacher_temp: float = 1.0
    distill_uniform_mix: float = 0.0
    # Confidence regularization: penalize normalized token entropy below a floor.
    entropy_floor_weight: float = 0.0
    entropy_floor_start: float = 0.0
    entropy_floor_end: float = 0.0
    # Loop-trajectory smoothness: penalize second differences of per-token entropy
    # across the loop axis. This is the main lightweight loop-curve stabilizer.
    entropy_curvature_weight: float = 0.0
    # Optional sampled logit-space curvature on a small set of uncertain tokens.
    # This is heavier than entropy curvature, so it is capped by max_positions.
    logit_curvature_weight: float = 0.0
    logit_curvature_max_positions: int = 0
    # Focus auxiliary regularizers on uncertain tokens using teacher entropy and
    # top1-top2 probability gap. Defaults preserve dense weighting.
    uncertainty_entropy_min: float = 0.0
    uncertainty_top2_gap_max: float = 1.0
    # Local hidden-state consistency across adjacent loops. Metric applies per token.
    local_consistency_weight: float = 0.0
    local_consistency_metric: Literal["cosine", "mse"] = "cosine"


@dataclass
class GRPOConfig:
    """Group Relative Policy Optimization (DeepSeekMath §4.1) configuration."""
    enabled: bool = False
    # Path to SFT checkpoint used as both π_θ_init and frozen π_ref.
    init_ckpt: str = ""
    # Prompts file: JSONL of {"prompt": str, "reference": str, "task": str}.
    prompts_file: str = ""
    # Group size: how many rollouts per prompt to form group-relative baseline.
    group_size: int = 8
    # Generation settings for rollouts (behavior policy π_θ_old).
    rollout_temperature: float = 1.0
    rollout_top_k: int = 64
    rollout_max_new_tokens: int = 512
    rollout_L: int = 4
    # PPO-style clip eps.
    clip_eps: float = 0.2
    # KL penalty coefficient β vs frozen π_ref (DeepSeek unbiased form).
    kl_beta: float = 0.05
    # Reuse old-policy logprobs for this many optimizer steps before resampling.
    # DeepSeek paper uses μ=1 (on-policy). Higher μ trades sample-efficiency for stability.
    mu_steps: int = 1
    # Verifier task (see TASK_VERIFIERS in verifiers.py).
    task: str = "gsm8k"
    # Optional reward-model checkpoint and reward mixing weights.
    reward_model_ckpt: str = ""
    reward_alpha: float = 0.3
    verifier_beta: float = 0.7
    prompt_budget: int = 10_000


@dataclass
class RewardModelConfig:
    """Pairwise preference training configuration."""
    enabled: bool = False
    init_ckpt: str = ""
    preferences_file: str = ""
    train_L: int = 4
    freeze_backbone: bool = False
    margin: float = 0.0


@dataclass
class OffloadConfig:
    """NVMe offload configuration."""
    enabled: bool = False
    root: str | None = None   # e.g., "H:/elt_data/offload_nvme"; if None, uses run_dir / "offload_nvme"
    # Safety: require at least this many free bytes before allocating NVMe state.
    min_free_gb: float = 20.0


@dataclass
class OptimConfig:
    """Optimizer selection. Phase-B adds 8-bit paged Adam; Phase-C adds NVMe-backed."""
    kind: Literal["adamw", "paged_adamw_8bit", "nvme_adamw"] = "adamw"
    # bitsandbytes.optim.PagedAdamW8bit / PagedAdamW32bit options.
    # percentile_clipping=100 disables BnB's internal clip (we use grad_clip externally).
    paged_percentile_clipping: int = 100
    # Optional: bits (8 or 32) for the paged optimizer. 8 => PagedAdamW8bit.
    paged_bits: Literal[8, 32] = 8


@dataclass
class DataConfig:
    train_bin: str = "data_bin/train.bin"      # uint32 packed token stream
    val_bin: str | None = "data_bin/val.bin"
    seq_len: int = 2048
    # Tokenizer directory (expect tokenizer.json + tokenizer_config.json).
    tokenizer_path: str = "H:/Qwen3.5-9B-official-hf"


@dataclass
class TrainConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    ilsd: ILSDConfig = field(default_factory=ILSDConfig)
    grpo: GRPOConfig = field(default_factory=GRPOConfig)
    reward_model: RewardModelConfig = field(default_factory=RewardModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    offload: OffloadConfig = field(default_factory=OffloadConfig)

    # Optimizer
    lr: float = 3e-4
    min_lr: float = 3e-5
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.96
    eps: float = 1e-8
    grad_clip: float = 1.0

    # Schedule
    warmup_steps: int = 2_000
    total_steps: int = 50_000
    lr_schedule: Literal["cosine", "linear", "constant"] = "cosine"

    # Batch
    micro_batch_size: int = 2
    grad_accum_steps: int = 16

    # Precision / memory
    dtype: Literal["bf16", "fp16", "fp32"] = "bf16"

    # Logging
    log_every: int = 10
    eval_every: int = 1_000
    save_every: int = 2_000
    run_dir: str = "runs/default"

    # Rolling checkpoint (crash recovery, independent of save_every milestones).
    # Saves to rolling_{0..keep-1}.pt round-robin; always updates last.pt hardlink.
    rolling_ckpt_interval_sec: int = 300   # 5 min
    rolling_ckpt_keep: int = 3

    # Reproducibility
    seed: int = 42

    @property
    def effective_batch_size(self) -> int:
        return self.micro_batch_size * self.grad_accum_steps


def load_train_config(path: str | Path) -> TrainConfig:
    """Load YAML into TrainConfig with nested dataclasses."""
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    model_raw = raw.pop("model", {}) or {}
    ilsd_raw = raw.pop("ilsd", {}) or {}
    grpo_raw = raw.pop("grpo", {}) or {}
    reward_model_raw = raw.pop("reward_model", {}) or {}
    data_raw = raw.pop("data", {}) or {}
    optim_raw = raw.pop("optim", {}) or {}
    offload_raw = raw.pop("offload", {}) or {}

    return TrainConfig(
        model=ModelConfig(**model_raw),
        ilsd=ILSDConfig(**ilsd_raw),
        grpo=GRPOConfig(**grpo_raw),
        reward_model=RewardModelConfig(**reward_model_raw),
        data=DataConfig(**data_raw),
        optim=OptimConfig(**optim_raw),
        offload=OffloadConfig(**offload_raw),
        **raw,
    )
