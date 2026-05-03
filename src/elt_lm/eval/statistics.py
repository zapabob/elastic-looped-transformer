"""Small statistical helpers for benchmark validation reports."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from statistics import mean, stdev
from typing import Mapping, Sequence


@dataclass(frozen=True)
class FoldStats:
    fold_count: int
    fold_scores: list[float]
    mean: float
    std: float
    sem: float
    ci95_low: float
    ci95_high: float


@dataclass(frozen=True)
class GroupSummary:
    name: str
    n: int
    mean: float
    std: float
    sem: float
    ci95_low: float
    ci95_high: float


@dataclass(frozen=True)
class PairwiseComparison:
    left: str
    right: str
    mean_delta: float
    p_value: float
    method: str


@dataclass(frozen=True)
class FriedmanComparison:
    groups: list[str]
    n_blocks: int
    statistic: float
    p_value: float
    method: str


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


def summarize_scores(name: str, scores: Sequence[float]) -> GroupSummary:
    values = [float(x) for x in scores]
    if not values:
        raise ValueError("summarize_scores requires at least one score")
    score_mean = mean(values)
    score_std = stdev(values) if len(values) > 1 else 0.0
    score_sem = score_std / math.sqrt(max(1, len(values)))
    ci = 1.96 * score_sem
    return GroupSummary(
        name=name,
        n=len(values),
        mean=score_mean,
        std=score_std,
        sem=score_sem,
        ci95_low=score_mean - ci,
        ci95_high=score_mean + ci,
    )


def _paired_values(left: Sequence[float], right: Sequence[float]) -> tuple[list[float], list[float]]:
    a = [float(x) for x in left]
    b = [float(x) for x in right]
    if len(a) != len(b):
        raise ValueError("paired comparisons require equal-length score arrays")
    if not a:
        raise ValueError("paired comparisons require at least one paired score")
    return a, b


def paired_permutation_pvalue(
    left: Sequence[float],
    right: Sequence[float],
    *,
    permutations: int = 10000,
    seed: int = 0,
) -> float:
    """Two-sided paired randomization p-value for mean difference."""

    a, b = _paired_values(left, right)
    diffs = [x - y for x, y in zip(a, b)]
    observed = abs(mean(diffs))
    if observed == 0.0:
        return 1.0
    rng = random.Random(seed)
    extreme = 1
    total = 1
    max_exact = 1 << len(diffs)
    if len(diffs) <= 16:
        iterator = range(max_exact)
        for mask in iterator:
            signed = [
                value if (mask >> idx) & 1 else -value
                for idx, value in enumerate(diffs)
            ]
            if abs(mean(signed)) >= observed - 1e-12:
                extreme += 1
            total += 1
    else:
        for _ in range(max(1, permutations)):
            signed = [value if rng.random() < 0.5 else -value for value in diffs]
            if abs(mean(signed)) >= observed - 1e-12:
                extreme += 1
            total += 1
    return min(1.0, extreme / total)


def _rank_block(values: Sequence[float]) -> list[float]:
    order = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and order[j][1] == order[i][1]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for idx in range(i, j):
            ranks[order[idx][0]] = avg_rank
        i = j
    return ranks


def friedman_statistic(groups: Mapping[str, Sequence[float]]) -> tuple[list[str], int, float]:
    names = list(groups)
    if len(names) < 3:
        raise ValueError("Friedman comparison requires at least three groups")
    arrays = [[float(x) for x in groups[name]] for name in names]
    n_blocks = len(arrays[0])
    if n_blocks == 0:
        raise ValueError("Friedman comparison requires at least one block")
    if any(len(values) != n_blocks for values in arrays):
        raise ValueError("all groups must have equal-length paired scores")
    k = len(names)
    rank_sums = [0.0] * k
    for block_idx in range(n_blocks):
        ranks = _rank_block([values[block_idx] for values in arrays])
        for group_idx, rank in enumerate(ranks):
            rank_sums[group_idx] += rank
    q = (12.0 / (n_blocks * k * (k + 1))) * sum(r * r for r in rank_sums)
    q -= 3.0 * n_blocks * (k + 1)
    return names, n_blocks, max(0.0, q)


def friedman_permutation_test(
    groups: Mapping[str, Sequence[float]],
    *,
    permutations: int = 10000,
    seed: int = 0,
) -> FriedmanComparison:
    """Permutation p-value for repeated-measures multi-group comparison."""

    names, n_blocks, observed = friedman_statistic(groups)
    arrays = [[float(x) for x in groups[name]] for name in names]
    rng = random.Random(seed)
    extreme = 1
    total = 1
    for _ in range(max(1, permutations)):
        shuffled: dict[str, list[float]] = {name: [] for name in names}
        for block_idx in range(n_blocks):
            block = [values[block_idx] for values in arrays]
            rng.shuffle(block)
            for name, value in zip(names, block):
                shuffled[name].append(value)
        _, _, stat = friedman_statistic(shuffled)
        if stat >= observed - 1e-12:
            extreme += 1
        total += 1
    return FriedmanComparison(
        groups=names,
        n_blocks=n_blocks,
        statistic=observed,
        p_value=min(1.0, extreme / total),
        method=f"friedman_within_block_permutation_{permutations}",
    )


def pairwise_group_comparisons(
    groups: Mapping[str, Sequence[float]],
    *,
    permutations: int = 10000,
    seed: int = 0,
) -> list[PairwiseComparison]:
    names = list(groups)
    comparisons: list[PairwiseComparison] = []
    for i, left in enumerate(names):
        for j, right in enumerate(names):
            if j <= i:
                continue
            a, b = _paired_values(groups[left], groups[right])
            comparisons.append(
                PairwiseComparison(
                    left=left,
                    right=right,
                    mean_delta=mean(a) - mean(b),
                    p_value=paired_permutation_pvalue(
                        a,
                        b,
                        permutations=permutations,
                        seed=seed + i * 1009 + j,
                    ),
                    method=f"paired_permutation_{permutations}",
                )
            )
    return comparisons
