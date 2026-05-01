"""Small statistical helpers for benchmark validation reports."""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import mean, stdev
from typing import Sequence


@dataclass(frozen=True)
class FoldStats:
    fold_count: int
    fold_scores: list[float]
    mean: float
    std: float
    sem: float
    ci95_low: float
    ci95_high: float


def fold_accuracy_stats(correct: Sequence[int | bool], *, folds: int) -> FoldStats:
    """Return deterministic K-fold accuracy summary for binary correctness."""

    values = [1 if bool(item) else 0 for item in correct]
    if not values:
        raise ValueError("fold_accuracy_stats requires at least one value")
    k = max(1, min(int(folds), len(values)))
    fold_scores: list[float] = []
    for fold_idx in range(k):
        fold_values = values[fold_idx::k]
        if not fold_values:
            continue
        fold_scores.append(sum(fold_values) / len(fold_values))
    score_mean = mean(fold_scores)
    score_std = stdev(fold_scores) if len(fold_scores) > 1 else 0.0
    score_sem = score_std / math.sqrt(max(1, len(fold_scores)))
    ci = 1.96 * score_sem
    return FoldStats(
        fold_count=len(fold_scores),
        fold_scores=fold_scores,
        mean=score_mean,
        std=score_std,
        sem=score_sem,
        ci95_low=max(0.0, score_mean - ci),
        ci95_high=min(1.0, score_mean + ci),
    )
