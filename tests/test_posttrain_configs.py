from __future__ import annotations

from pathlib import Path

from elt_lm.config import load_train_config


def test_load_train_config_parses_reward_model_and_grpo_extensions(tmp_path: Path) -> None:
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        """
model:
  d_model: 32
  n_unique_layers: 2
  n_heads: 2
  d_ff: 64
grpo:
  enabled: true
  init_ckpt: runs/sft.pt
  prompts_file: prompts.jsonl
  reward_model_ckpt: runs/rm.pt
  reward_alpha: 0.25
  verifier_beta: 0.75
  prompt_budget: 123
reward_model:
  enabled: true
  init_ckpt: runs/base.pt
  preferences_file: prefs.jsonl
  train_L: 3
  freeze_backbone: true
        """,
        encoding="utf-8",
    )
    cfg = load_train_config(cfg_path)
    assert cfg.grpo.reward_model_ckpt == "runs/rm.pt"
    assert abs(cfg.grpo.reward_alpha - 0.25) < 1e-6
    assert abs(cfg.grpo.verifier_beta - 0.75) < 1e-6
    assert cfg.grpo.prompt_budget == 123
    assert cfg.reward_model.enabled is True
    assert cfg.reward_model.preferences_file == "prefs.jsonl"
    assert cfg.reward_model.train_L == 3
    assert cfg.reward_model.freeze_backbone is True
