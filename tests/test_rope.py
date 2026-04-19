"""RoPE numerical correctness tests."""

from __future__ import annotations

import math

import torch

from elt_lm.rope import RoPECache, apply_rope


def test_rope_cache_shape() -> None:
    cache = RoPECache(head_dim=64, max_seq_len=128, theta=10000.0)
    cos, sin = cache(32, device=torch.device("cpu"), dtype=torch.float32)
    assert cos.shape == (32, 64)
    assert sin.shape == (32, 64)


def test_rope_identity_at_position_zero() -> None:
    """At position 0, cos=1, sin=0, so q and k should pass through unchanged."""
    B, H, T, D = 2, 3, 1, 8
    q = torch.randn(B, H, T, D)
    k = torch.randn(B, H, T, D)
    cache = RoPECache(head_dim=D, max_seq_len=16)
    cos, sin = cache(T, device=q.device, dtype=q.dtype)
    q_r, k_r = apply_rope(q, k, cos, sin)
    assert torch.allclose(q_r, q, atol=1e-6)
    assert torch.allclose(k_r, k, atol=1e-6)


def test_rope_preserves_norm() -> None:
    """Rotation is orthogonal -> 2-norm is preserved per head."""
    B, H, T, D = 2, 4, 8, 16
    torch.manual_seed(0)
    q = torch.randn(B, H, T, D)
    k = torch.randn(B, H, T, D)
    cache = RoPECache(head_dim=D, max_seq_len=16)
    cos, sin = cache(T, device=q.device, dtype=q.dtype)
    q_r, k_r = apply_rope(q, k, cos, sin)

    q_norm_before = q.pow(2).sum(-1).sqrt()
    q_norm_after = q_r.pow(2).sum(-1).sqrt()
    k_norm_before = k.pow(2).sum(-1).sqrt()
    k_norm_after = k_r.pow(2).sum(-1).sqrt()
    assert torch.allclose(q_norm_before, q_norm_after, atol=1e-5)
    assert torch.allclose(k_norm_before, k_norm_after, atol=1e-5)


def test_rope_frequency_spectrum() -> None:
    """Verify the lowest frequency inv_freq = 1, highest = 1/theta^((D-2)/D)."""
    D = 8
    theta = 10000.0
    cache = RoPECache(head_dim=D, max_seq_len=4, theta=theta)
    # At t=1, cos[1, 0] = cos(1 * 1/theta^0) = cos(1)
    cos, _sin = cache(4, device=torch.device("cpu"), dtype=torch.float32)
    assert math.isclose(float(cos[1, 0]), math.cos(1.0), abs_tol=1e-6)
