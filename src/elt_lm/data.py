"""Packed token-stream dataset backed by a memory-mapped uint32 file."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class PackedTokenDataset(Dataset):
    """Dataset over a flat uint32 token stream.

    Each sample is a window of (seq_len + 1) tokens, split into:
        input_ids = window[:seq_len]
        labels    = window[1:seq_len + 1]

    Windows are non-overlapping by default (stride = seq_len). This is the standard
    next-token-prediction setup — shift-CE in the loss consumes the extra +1 token.
    """

    def __init__(self, bin_path: str | Path, seq_len: int, stride: int | None = None):
        self.path = Path(bin_path)
        if not self.path.is_file():
            raise FileNotFoundError(f"token bin not found: {self.path}")
        self.seq_len = seq_len
        self.stride = stride if stride is not None else seq_len

        # mmap as uint32 read-only
        self._array = np.memmap(self.path, dtype=np.uint32, mode="r")
        self.n_tokens = int(self._array.shape[0])

        # Number of full windows we can produce.
        last_start = self.n_tokens - (self.seq_len + 1)
        if last_start < 0:
            raise ValueError(
                f"bin has only {self.n_tokens} tokens; need at least {self.seq_len + 1}"
            )
        self._n_windows = last_start // self.stride + 1

    def __len__(self) -> int:
        return self._n_windows

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        if idx < 0:
            idx += self._n_windows
        if idx < 0 or idx >= self._n_windows:
            raise IndexError(idx)
        start = idx * self.stride
        window = self._array[start: start + self.seq_len + 1].astype(np.int64, copy=True)
        input_ids = torch.from_numpy(window[:-1])       # (seq_len,)
        labels = torch.from_numpy(window).clone()       # we use shift-CE in ilsd._causal_lm_ce
        labels = labels[: self.seq_len]                 # align back to (seq_len,)
        # Note: the shift-by-1 happens inside _causal_lm_ce, so here both are (seq_len,)
        # and labels is just a copy of input_ids (next-token target is derived there).
        # Simpler convention:
        return input_ids, labels
