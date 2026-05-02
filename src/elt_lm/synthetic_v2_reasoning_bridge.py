"""Build bridge/easy-hard GRPO prompts for synthetic v2 math and STEM.

The hard v2 reasoning prompts are intentionally multi-step, but the first live
GRPO run produced no verifier positives for math and STEM.  This module creates
bridge curricula for those lanes without replacing the completed hard-run
evidence:

* easy prompts that should be reachable by the current SFT adapters,
* bridge prompts that retain two-step traps with simpler arithmetic/causality,
* a retained slice of the original hard prompts.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from fractions import Fraction
import json
from pathlib import Path
from typing import Any, Literal


Lane = Literal["math", "stem_reasoning"]

SOURCE_NAME = "synthetic-v2-reasoning-bridge"
LANES: tuple[Lane, ...] = ("math", "stem_reasoning")
DEFAULT_HARD_CASES: dict[Lane, Path] = {
    "math": Path("H:/elt_data/synthetic_v2_hard/math/benchmarks/synthetic_v2_hard_math_val_cases.jsonl"),
    "stem_reasoning": Path(
        "H:/elt_data/synthetic_v2_hard/stem_reasoning/benchmarks/"
        "synthetic_v2_hard_stem_reasoning_val_cases.jsonl"
    ),
}
DEFAULT_OUTPUTS: dict[Lane, Path] = {
    "math": Path("H:/elt_data/synthetic_v2_hard/math/benchmarks/synthetic_v2_bridge_math_val_cases.jsonl"),
    "stem_reasoning": Path(
        "H:/elt_data/synthetic_v2_hard/stem_reasoning/benchmarks/"
        "synthetic_v2_bridge_stem_reasoning_val_cases.jsonl"
    ),
}
TASKS: dict[Lane, str] = {
    "math": "exact_math",
    "stem_reasoning": "mcq_reasoning",
}
BUCKETS: dict[Lane, str] = {
    "math": "gguf_math_distill_v2_bridge",
    "stem_reasoning": "gguf_stem_reasoning_distill_v2_bridge",
}


def _frac(value: Fraction) -> str:
    if value.denominator == 1:
        return str(value.numerator)
    return f"{value.numerator}/{value.denominator}"


def _response(answer: str, reason: str) -> str:
    return f"<think>{reason}</think><answer>{answer}</answer>"


def _math_prompt(problem: str, idx: int, difficulty: str) -> str:
    return (
        "Solve the following math problem.\n"
        "Return your result using <think>...</think><answer>...</answer>.\n"
        "Put only the final exact answer inside <answer>.\n\n"
        "Problem:\n"
        f"{problem} Synthetic v2 bridge id {idx} ({difficulty})."
    )


def _stem_prompt(question: str, choices: list[str], idx: int, difficulty: str) -> str:
    return (
        "Answer the following STEM multiple-choice question.\n"
        "Return your result using <think>...</think><answer>...</answer>.\n"
        "Put only the final option letter inside <answer>.\n\n"
        "Question:\n"
        f"{question} Synthetic v2 bridge id {idx} ({difficulty}).\n\n"
        "Choices:\n"
        + "\n".join(choices)
    )


def _choices(correct: str, correct_text: str, distractors: list[str]) -> list[str]:
    out: list[str] = []
    distractor_iter = iter(distractors)
    for letter in "ABCD":
        text = correct_text if letter == correct else next(distractor_iter)
        out.append(f"{letter}. {text}")
    return out


@dataclass(frozen=True)
class ReasoningBridgePrompt:
    lane: Lane
    prompt: str
    reference: str
    correct_response: str
    difficulty: str
    domain: str
    idx: int

    def to_record(self) -> dict[str, Any]:
        return {
            "prompt": self.prompt,
            "reference": self.reference,
            "task": TASKS[self.lane],
            "bucket": BUCKETS[self.lane],
            "source": SOURCE_NAME,
            "metadata": {
                "lane": self.lane,
                "task_name": self.domain,
                "difficulty": self.difficulty,
                "curriculum": "bridge_easy_hard",
                "variant": f"synthetic_v2_bridge_{self.lane}_{self.difficulty}_{self.idx}",
                "tags": [
                    self.lane,
                    "synthetic_v2_bridge",
                    self.difficulty,
                    TASKS[self.lane],
                ],
            },
        }


def _easy_math_prompt(idx: int) -> ReasoningBridgePrompt:
    kind = idx % 6
    if kind == 0:
        units = 3 + (idx % 3)
        final = 10 + units * 4 + 2
        problem = f"A job has base cost 10, {units} units at 4 each, and audit fee 2. What exact cost is charged?"
        reason = f"Compute the unit subtotal first: {units}*4={units * 4}. Add base and audit fee: 10+{units * 4}+2={final}."
        answer = str(final)
        domain = "easy_cost_arithmetic"
    elif kind == 1:
        total = 18 + 3 * (idx % 3)
        rejected = total // 3
        audited = (total - rejected) // 2
        problem = (
            f"Out of {total} records, one third are rejected. Half of the remaining records are audited. "
            "How many records are audited?"
        )
        reason = f"Reject {total}/3={rejected}, leaving {total - rejected}. Half of that is {audited}."
        answer = str(audited)
        domain = "easy_fraction_counts"
    elif kind == 2:
        x0 = 2 + (idx % 4)
        after = x0 + 3 + 3
        final = after * 2
        problem = f"A loop starts at x={x0}. Add 3 in each of two loops, then double the state. What integer results?"
        reason = f"After two additions the state is {x0}+3+3={after}. Doubling gives {final}."
        answer = str(final)
        domain = "easy_two_loop_state"
    elif kind == 3:
        a = 4 + (idx % 3)
        b = 8 + (idx % 3)
        final_frac = Fraction(a + 3 * b, 4)
        problem = f"Values {a} and {b} have weights 1 and 3. What exact weighted mean do they produce?"
        reason = f"The weighted numerator is {a}+3*{b}={a + 3 * b}, and total weight is 4, so the mean is {_frac(final_frac)}."
        answer = _frac(final_frac)
        domain = "easy_weighted_mean"
    elif kind == 4:
        pass_rate = Fraction(1, 2)
        pass_escalate = Fraction(3, 5)
        fail_escalate = Fraction(1, 10)
        final_frac = pass_rate * pass_escalate + (1 - pass_rate) * fail_escalate
        problem = (
            "A case passes a screen with probability 1/2. Passed cases escalate with probability 3/5, "
            "failed cases escalate with probability 1/10. What exact escalation probability results?"
        )
        reason = f"Condition on the screen: 1/2*3/5 + 1/2*1/10 = {_frac(final_frac)}."
        answer = _frac(final_frac)
        domain = "easy_conditioned_probability"
    else:
        first = 5 + (idx % 5)
        second = first + 4
        final = second - 2
        problem = f"A state starts at {first}, then adds 4 and finally subtracts 2. What integer is read out?"
        reason = f"Apply the operations in order: {first}+4={second}, then {second}-2={final}."
        answer = str(final)
        domain = "easy_ordered_operations"
    return ReasoningBridgePrompt(
        "math",
        _math_prompt(problem, idx, "easy"),
        answer,
        _response(answer, reason),
        "easy",
        domain,
        idx,
    )


def _bridge_math_prompt(idx: int) -> ReasoningBridgePrompt:
    kind = idx % 6
    if kind == 0:
        prior = Fraction(1, 5)
        hit = Fraction(3, 4)
        false = Fraction(1, 4)
        posterior = prior * hit / (prior * hit + (1 - prior) * false)
        final_frac = posterior * Fraction(2, 3) + (1 - posterior) * Fraction(1, 4)
        problem = (
            "An incident has prior probability 1/5. A detector fires with probability 3/4 during an incident "
            "and 1/4 otherwise. After a fire, incidents escalate with probability 2/3 and non-incidents with "
            "probability 1/4. What exact escalation probability follows after the detector fire?"
        )
        reason = f"Bayes gives P(incident|fire)={_frac(posterior)}. Then weight escalation: {_frac(posterior)}*2/3 + (1-{_frac(posterior)})*1/4 = {_frac(final_frac)}."
        answer = _frac(final_frac)
        domain = "bridge_bayes_then_escalation"
    elif kind == 1:
        x0 = 3 + (idx % 3)
        trace = [x0]
        x = x0
        for _ in range(3):
            x = 2 * x + 1
            trace.append(x)
        final = trace[-1] - 4
        problem = f"A loop state starts at {x0}. Each of three loops applies x <- 2x + 1. Then subtract 4. What integer is read out?"
        reason = f"Unroll the states as {trace}; the third state is {trace[-1]}, and subtracting 4 gives {final}."
        answer = str(final)
        domain = "bridge_recurrence_correction"
    elif kind == 2:
        total = 40 + 2 * (idx % 4)
        a = 15 + (idx % 3)
        b = 18 + (idx % 4)
        both = 5
        union = a + b - both
        final_frac = Fraction(union, total) * Fraction(1, 2)
        problem = (
            f"In a population of {total}, set A has {a}, set B has {b}, and the overlap has {both}. "
            "A second property occurs for half of A union B and never outside it. What exact probability has the second property?"
        )
        reason = f"Inclusion-exclusion gives |A union B|={a}+{b}-{both}={union}. Multiplying {union}/{total} by 1/2 gives {_frac(final_frac)}."
        answer = _frac(final_frac)
        domain = "bridge_inclusion_then_probability"
    elif kind == 3:
        units = 10 + (idx % 3)
        subtotal = 12 + 4 * 3 + max(0, units - 4) * 5
        final_frac = Fraction(subtotal) * Fraction(4, 5) + 3
        problem = (
            f"A metered task has base cost 12. The first 4 units cost 3 each and remaining units cost 5 each. "
            f"For {units} units, apply a 20% discount before adding audit fee 3. What exact final cost is charged?"
        )
        reason = f"Tiered subtotal is 12+4*3+({units}-4)*5={subtotal}. Apply the discount before the fee: {subtotal}*4/5+3={_frac(final_frac)}."
        answer = _frac(final_frac)
        domain = "bridge_tier_discount_fee"
    elif kind == 4:
        initial = Fraction(1, 3)
        stay = Fraction(2, 3)
        recover = Fraction(1, 6)
        p1 = initial * stay + (1 - initial) * recover
        p2 = p1 * stay + (1 - p1) * recover
        final_frac = p2 * Fraction(3, 5)
        problem = (
            "A service starts unhealthy with probability 1/3. Each loop keeps an unhealthy service unhealthy "
            "with probability 2/3, while a healthy service becomes unhealthy with probability 1/6. After two loops, "
            "an audit catches an unhealthy service with probability 3/5. What exact probability does the audit catch it?"
        )
        reason = f"After one loop p1={_frac(p1)} and after the second p2={_frac(p2)}. Multiplying by 3/5 gives {_frac(final_frac)}."
        answer = _frac(final_frac)
        domain = "bridge_two_loop_markov"
    else:
        scores = [3 + (idx % 3), 6, 5, 8]
        weights = [1, 2, 3, 4]
        dropped = scores.index(min(scores))
        kept = [(s, w) for j, (s, w) in enumerate(zip(scores, weights)) if j != dropped]
        numerator = sum(s * w for s, w in kept)
        denominator = sum(w for _s, w in kept)
        final_frac = Fraction(numerator, denominator) * Fraction(3, 2)
        problem = (
            f"Scores {scores} have weights {weights}. Drop the single lowest score, compute the weighted mean "
            "of the remaining entries, then multiply by 3/2. What exact value is read out?"
        )
        reason = f"Drop index {dropped}; kept weighted numerator is {numerator} and kept weight is {denominator}. Applying gain 3/2 gives {_frac(final_frac)}."
        answer = _frac(final_frac)
        domain = "bridge_drop_low_weighted_gain"
    return ReasoningBridgePrompt(
        "math",
        _math_prompt(problem, idx, "bridge"),
        answer,
        _response(answer, reason),
        "bridge",
        domain,
        idx,
    )


def _easy_stem_prompt(idx: int) -> ReasoningBridgePrompt:
    correct = "ABCD"[idx % 4]
    kind = idx % 4
    if kind == 0:
        question = "A sealed box gets warmer after work is done on a paddle inside it. Which statement best explains the energy change?"
        correct_text = "Work transfers energy into the box, increasing internal energy while conserving total energy."
        distractors = [
            "Energy is created because temperature increased.",
            "The box must lose energy whenever work is done on it.",
            "Temperature can change only when mass enters or leaves.",
        ]
        domain = "easy_energy_conservation"
    elif kind == 1:
        question = "A queue has short bursts that exceed its buffer even though the long-run average input is low. What is the best systems explanation?"
        correct_text = "Finite buffers can overflow during bursts, so average rate alone is not a sufficient guarantee."
        distractors = [
            "An average input below service rate makes overflow impossible.",
            "Buffers overflow only when service rate is exactly zero.",
            "Burstiness is irrelevant once the mean is known.",
        ]
        domain = "easy_queue_bursts"
    elif kind == 2:
        question = "A rare disease test has high sensitivity but imperfect specificity. Why can many positive results still be false positives?"
        correct_text = "Low prevalence leaves many non-cases, so imperfect specificity can dominate the positive pool."
        distractors = [
            "Sensitivity alone determines the positive predictive value.",
            "Specificity affects only negative results.",
            "Prevalence stops mattering after any positive result.",
        ]
        domain = "easy_prevalence_specificity"
    else:
        question = "A controller reduces error only after a delayed sensor update. Why can a too-large gain cause oscillation?"
        correct_text = "The delayed feedback can overcorrect the old error, pushing the system past the target."
        distractors = [
            "Feedback delay always makes systems perfectly stable.",
            "Higher gain must reduce oscillation in every feedback system.",
            "Oscillation requires random noise and cannot come from delay.",
        ]
        domain = "easy_delayed_feedback"
    reason = "Check the governing constraint first, then apply the second condition rather than using a one-factor shortcut."
    choices = _choices(correct, correct_text, distractors)
    return ReasoningBridgePrompt(
        "stem_reasoning",
        _stem_prompt(question, choices, idx, "easy"),
        correct,
        _response(correct, reason),
        "easy",
        domain,
        idx,
    )


def _bridge_stem_prompt(idx: int) -> ReasoningBridgePrompt:
    correct = "ABCD"[idx % 4]
    kind = idx % 5
    if kind == 0:
        question = (
            "A battery pack runs cooler after airflow is redirected across only the hottest cells, even though total fan power is unchanged. "
            "Which interpretation best separates energy conservation from thermal-risk reduction?"
        )
        correct_text = "Total energy is still conserved; targeted airflow changes heat removal where the limiting temperature occurs."
        distractors = [
            "Cooler cells prove the pack created less heat without changing load.",
            "Fan power alone determines every cell temperature, so airflow placement cannot matter.",
            "Thermal risk must rise whenever heat is moved away from any one cell.",
        ]
        domain = "bridge_battery_thermal_path"
    elif kind == 1:
        question = (
            "A lab assay has high sensitivity and moderate specificity. In a low-prevalence screening population, confirmatory tests remove many positives. "
            "Which conclusion follows?"
        )
        correct_text = "The first positive pool can be dominated by false positives, so confirmation improves precision by conditioning again."
        distractors = [
            "High sensitivity guarantees that most first positives are true positives.",
            "Confirmatory tests matter only for negative samples.",
            "Prevalence is irrelevant once the assay is sensitive.",
        ]
        domain = "bridge_assay_confirmation"
    elif kind == 2:
        question = (
            "A reservoir controller keeps the mean inflow below mean outflow, but overflow happens when rain arrives in correlated bursts. "
            "Which explanation is strongest?"
        )
        correct_text = "Stability by means is insufficient; burst correlation and finite storage determine overflow risk."
        distractors = [
            "Mean outflow above mean inflow mathematically forbids overflow.",
            "Correlation affects only measurement error, not storage.",
            "Finite storage behaves like infinite storage if the average is safe.",
        ]
        domain = "bridge_reservoir_bursts"
    elif kind == 3:
        question = (
            "An enzyme pathway speeds up after a cofactor is added, but only until substrate transport becomes limiting. "
            "Which interpretation best fits the two constraints?"
        )
        correct_text = "The cofactor improves one step, then the bottleneck shifts to transport, limiting further rate gains."
        distractors = [
            "Adding cofactor must keep increasing rate without bound.",
            "Transport cannot become limiting if the enzyme is faster.",
            "A plateau means the cofactor had no effect at any point.",
        ]
        domain = "bridge_pathway_bottleneck_shift"
    else:
        question = (
            "A sensor fusion system improves recall after adding a noisy secondary sensor, but precision falls unless a second-stage filter is used. "
            "What is the best explanation?"
        )
        correct_text = "The secondary sensor catches more true events and more false alarms; filtering changes the precision-recall tradeoff."
        distractors = [
            "Any added sensor must improve recall and precision equally.",
            "Precision cannot change if recall changes.",
            "A second-stage filter affects only sensor latency, not classification outcomes.",
        ]
        domain = "bridge_sensor_fusion_tradeoff"
    reason = "Reject the tempting single-metric answer and trace both the first-stage effect and the limiting or filtering constraint."
    choices = _choices(correct, correct_text, distractors)
    return ReasoningBridgePrompt(
        "stem_reasoning",
        _stem_prompt(question, choices, idx, "bridge"),
        correct,
        _response(correct, reason),
        "bridge",
        domain,
        idx,
    )


def generate_easy_reasoning_bridge_prompts(lane: Lane, count: int) -> list[ReasoningBridgePrompt]:
    if lane == "math":
        return [_easy_math_prompt(i) for i in range(count)]
    if lane == "stem_reasoning":
        return [_easy_stem_prompt(i) for i in range(count)]
    raise ValueError(f"unsupported lane: {lane}")


def generate_bridge_reasoning_prompts(lane: Lane, count: int) -> list[ReasoningBridgePrompt]:
    if lane == "math":
        return [_bridge_math_prompt(i) for i in range(count)]
    if lane == "stem_reasoning":
        return [_bridge_stem_prompt(i) for i in range(count)]
    raise ValueError(f"unsupported lane: {lane}")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _hard_records(lane: Lane, path: Path, count: int) -> list[dict[str, Any]]:
    rows = _read_jsonl(path)
    if not rows:
        raise RuntimeError(f"no hard prompt records in {path}")
    out: list[dict[str, Any]] = []
    for i in range(count):
        row = dict(rows[i % len(rows)])
        metadata = dict(row.get("metadata") or {})
        metadata.update({
            "difficulty": "hard",
            "curriculum": "bridge_easy_hard",
            "source_cases": str(path),
        })
        tags = list(metadata.get("tags") or [])
        for tag in ("synthetic_v2_bridge", "hard"):
            if tag not in tags:
                tags.append(tag)
        metadata["tags"] = tags
        row["metadata"] = metadata
        row["source"] = SOURCE_NAME
        row["bucket"] = BUCKETS[lane]
        out.append(row)
    return out


def build_lane_bridge_prompts(
    *,
    lane: Lane,
    output_path: Path | None = None,
    hard_cases_path: Path | None = None,
    total_cases: int = 256,
    easy_cases: int | None = None,
    bridge_cases: int | None = None,
) -> dict[str, Any]:
    if total_cases <= 0:
        raise ValueError("total_cases must be positive")
    output = output_path or DEFAULT_OUTPUTS[lane]
    hard_path = hard_cases_path or DEFAULT_HARD_CASES[lane]
    easy_n = easy_cases if easy_cases is not None else max(1, total_cases // 4)
    bridge_n = bridge_cases if bridge_cases is not None else max(1, total_cases // 2)
    hard_n = total_cases - easy_n - bridge_n
    if min(easy_n, bridge_n, hard_n) <= 0:
        raise ValueError("easy, bridge, and hard case counts must all be positive")

    easy = [item.to_record() for item in generate_easy_reasoning_bridge_prompts(lane, easy_n)]
    bridge = [item.to_record() for item in generate_bridge_reasoning_prompts(lane, bridge_n)]
    hard = _hard_records(lane, hard_path, hard_n)

    max_len = max(len(easy), len(bridge), len(hard))
    mixed: list[dict[str, Any]] = []
    for i in range(max_len):
        if i < len(bridge):
            mixed.append(bridge[i])
        if i < len(easy):
            mixed.append(easy[i])
        if i < len(hard):
            mixed.append(hard[i])
        if i + len(hard) < len(bridge):
            mixed.append(bridge[i + len(hard)])
    mixed = mixed[:total_cases]

    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8", newline="\n") as f:
        for row in mixed:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    counts: dict[str, int] = {"easy": 0, "bridge": 0, "hard": 0}
    domains: dict[str, int] = {}
    for row in mixed:
        md = row.get("metadata") or {}
        difficulty = str(md.get("difficulty", "unknown"))
        counts[difficulty] = counts.get(difficulty, 0) + 1
        domain = str(md.get("task_name", "unknown"))
        domains[domain] = domains.get(domain, 0) + 1
    summary = {
        "source": SOURCE_NAME,
        "lane": lane,
        "output_path": str(output),
        "hard_cases_path": str(hard_path),
        "total_cases": len(mixed),
        "difficulty_counts": counts,
        "domain_counts": domains,
    }
    output.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def cli() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lanes", nargs="+", choices=LANES, default=list(LANES))
    parser.add_argument("--total-cases", type=int, default=256)
    parser.add_argument("--easy-cases", type=int, default=None)
    parser.add_argument("--bridge-cases", type=int, default=None)
    parser.add_argument("--math-output", type=Path, default=DEFAULT_OUTPUTS["math"])
    parser.add_argument("--stem-output", type=Path, default=DEFAULT_OUTPUTS["stem_reasoning"])
    parser.add_argument("--math-hard-cases", type=Path, default=DEFAULT_HARD_CASES["math"])
    parser.add_argument("--stem-hard-cases", type=Path, default=DEFAULT_HARD_CASES["stem_reasoning"])
    args = parser.parse_args()

    outputs: dict[Lane, Path] = {
        "math": args.math_output,
        "stem_reasoning": args.stem_output,
    }
    hard_cases: dict[Lane, Path] = {
        "math": args.math_hard_cases,
        "stem_reasoning": args.stem_hard_cases,
    }
    summaries = [
        build_lane_bridge_prompts(
            lane=lane,
            output_path=outputs[lane],
            hard_cases_path=hard_cases[lane],
            total_cases=args.total_cases,
            easy_cases=args.easy_cases,
            bridge_cases=args.bridge_cases,
        )
        for lane in args.lanes
    ]
    print(json.dumps({"summaries": summaries}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    cli()
