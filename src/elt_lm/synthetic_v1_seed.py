from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .gguf_distill import (
    DistillQualityError,
    DistillTask,
    build_sft_record,
    evaluate_distill_records,
    validate_distill_record_quality,
)


LANES: tuple[str, ...] = ("code", "math", "stem_reasoning", "tool_use")


@dataclass(frozen=True)
class SyntheticExample:
    task: DistillTask
    example: dict[str, Any]


def _task(lane: str, name: str, description: str, target_kind: str, index: int) -> DistillTask:
    return DistillTask(
        lane=lane,  # type: ignore[arg-type]
        domain=name,
        description=description,
        target_kind=target_kind,
        tags=[lane, "synthetic_v1_seed"],
        target_label="",
        risk_tags=[],
        variant_index=index,
        mode="synthetic",
        variant=f"synthetic_seed_{index}",
    )


def _code_examples(count: int) -> Iterable[SyntheticExample]:
    def clamp(i: int) -> tuple[str, str, str, str]:
        low = i % 7
        high = low + 5 + (i % 4)
        below = low - 3
        above = high + 4
        return (
            "clamp_int",
            f"Implement clamp_value(value, low, high) with typed integer bounds. Seed case: {i}.",
            "def clamp_value(value: int, low: int, high: int) -> int:\n"
            "    if low > high:\n"
            "        raise ValueError('low must be <= high')\n"
            "    return min(max(value, low), high)",
            f"assert clamp_value({low + 2}, {low}, {high}) == {low + 2}\n"
            f"assert clamp_value({below}, {low}, {high}) == {low}\n"
            f"assert clamp_value({above}, {low}, {high}) == {high}\n"
            "try:\n"
            f"    clamp_value(1, {high}, {low})\n"
            "except ValueError as exc:\n"
            "    assert str(exc) == 'low must be <= high'\n"
            "else:\n"
            "    raise AssertionError('expected ValueError')",
        )

    def normalize(i: int) -> tuple[str, str, str, str]:
        a = f"Tag{i % 11}"
        b = f"Group{(i * 3) % 13}"
        return (
            "normalize_tags",
            f"Implement normalize_tags(tags) returning sorted lowercase unique non-empty tags. Seed case: {i}.",
            "def normalize_tags(tags: list[str]) -> list[str]:\n"
            "    cleaned = {tag.strip().lower() for tag in tags if tag.strip()}\n"
            "    return sorted(cleaned)",
            f"assert normalize_tags([' {a} ', '{a.lower()}', '', '{b}']) == ['{b.lower()}', '{a.lower()}']\n"
            "assert normalize_tags([]) == []\n"
            f"assert normalize_tags(['Z{i}', 'a{i}', 'z{i}']) == ['a{i}', 'z{i}']",
        )

    def parse_kv(i: int) -> tuple[str, str, str, str]:
        key = f"mode_{i % 17}"
        value = f"safe_{(i * 5) % 19}"
        return (
            "parse_kv_line",
            f"Implement parse_kv_line(line) for 'key=value' pairs with trimmed fields. Seed case: {i}.",
            "def parse_kv_line(line: str) -> tuple[str, str]:\n"
            "    if '=' not in line:\n"
            "        raise ValueError('missing separator')\n"
            "    key, value = line.split('=', 1)\n"
            "    key = key.strip()\n"
            "    value = value.strip()\n"
            "    if not key:\n"
            "        raise ValueError('empty key')\n"
            "    return key, value",
            f"assert parse_kv_line(' {key} = {value} ') == ('{key}', '{value}')\n"
            f"assert parse_kv_line('{key}=a=b') == ('{key}', 'a=b')\n"
            "try:\n"
            "    parse_kv_line('novalue')\n"
            "except ValueError as exc:\n"
            "    assert str(exc) == 'missing separator'\n"
            "else:\n"
            "    raise AssertionError('expected missing separator')",
        )

    def rolling(i: int) -> tuple[str, str, str, str]:
        start = 1 + (i % 5)
        values = [float(start + step) for step in range(4)]
        expected = [round(sum(values[j:j + 2]) / 2, 3) for j in range(3)]
        return (
            "rolling_average",
            f"Implement rolling_average(values, window) returning rounded moving averages. Seed case: {i}.",
            "def rolling_average(values: list[float], window: int) -> list[float]:\n"
            "    if window <= 0:\n"
            "        raise ValueError('window must be positive')\n"
            "    if window > len(values):\n"
            "        return []\n"
            "    return [round(sum(values[i:i + window]) / window, 3) for i in range(len(values) - window + 1)]",
            f"assert rolling_average({values!r}, 2) == {expected!r}\n"
            f"assert rolling_average([{float(start)}], 2) == []\n"
            "try:\n"
            "    rolling_average([1.0], 0)\n"
            "except ValueError as exc:\n"
            "    assert str(exc) == 'window must be positive'\n"
            "else:\n"
            "    raise AssertionError('expected ValueError')",
        )

    def chunk_list(i: int) -> tuple[str, str, str, str]:
        size = 2 + (i % 3)
        values = list(range(i % 4, i % 4 + 7))
        chunks = [values[j:j + size] for j in range(0, len(values), size)]
        return (
            "chunk_list",
            f"Implement chunk_list(values, size) for typed list chunking with validation. Seed case: {i}.",
            "def chunk_list(values: list[int], size: int) -> list[list[int]]:\n"
            "    if size <= 0:\n"
            "        raise ValueError('size must be positive')\n"
            "    return [values[i:i + size] for i in range(0, len(values), size)]",
            f"assert chunk_list({values!r}, {size}) == {chunks!r}\n"
            "assert chunk_list([], 3) == []\n"
            "try:\n"
            "    chunk_list([1], 0)\n"
            "except ValueError as exc:\n"
            "    assert str(exc) == 'size must be positive'\n"
            "else:\n"
            "    raise AssertionError('expected ValueError')",
        )

    def safe_ratio(i: int) -> tuple[str, str, str, str]:
        numerator = 10 + i
        denominator = 2 + (i % 5)
        expected = round(numerator / denominator, 4)
        return (
            "safe_ratio",
            f"Implement safe_ratio(numerator, denominator) returning a rounded float and rejecting zero. Seed case: {i}.",
            "def safe_ratio(numerator: float, denominator: float) -> float:\n"
            "    if denominator == 0:\n"
            "        raise ValueError('denominator must be nonzero')\n"
            "    return round(numerator / denominator, 4)",
            f"assert safe_ratio({float(numerator)!r}, {float(denominator)!r}) == {expected!r}\n"
            "assert safe_ratio(0.0, 3.0) == 0.0\n"
            "try:\n"
            "    safe_ratio(1.0, 0.0)\n"
            "except ValueError as exc:\n"
            "    assert str(exc) == 'denominator must be nonzero'\n"
            "else:\n"
            "    raise AssertionError('expected ValueError')",
        )

    def merge_intervals(i: int) -> tuple[str, str, str, str]:
        base = i % 6
        intervals = [(base, base + 2), (base + 1, base + 4), (base + 6, base + 7)]
        expected = [(base, base + 4), (base + 6, base + 7)]
        return (
            "merge_intervals",
            f"Implement merge_intervals(intervals) for closed integer intervals. Seed case: {i}.",
            "def merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:\n"
            "    ordered = sorted(intervals)\n"
            "    merged: list[tuple[int, int]] = []\n"
            "    for start, end in ordered:\n"
            "        if start > end:\n"
            "            raise ValueError('invalid interval')\n"
            "        if not merged or start > merged[-1][1]:\n"
            "            merged.append((start, end))\n"
            "        else:\n"
            "            merged[-1] = (merged[-1][0], max(merged[-1][1], end))\n"
            "    return merged",
            f"assert merge_intervals({intervals!r}) == {expected!r}\n"
            "assert merge_intervals([]) == []\n"
            "try:\n"
            "    merge_intervals([(3, 2)])\n"
            "except ValueError as exc:\n"
            "    assert str(exc) == 'invalid interval'\n"
            "else:\n"
            "    raise AssertionError('expected ValueError')",
        )

    def topo_sort(i: int) -> tuple[str, str, str, str]:
        a = f"extract_{i % 5}"
        b = f"transform_{i % 7}"
        c = f"load_{i % 11}"
        return (
            "topological_sort",
            f"Implement topo_sort(edges) returning a deterministic dependency order. Seed case: {i}.",
            "from collections import defaultdict\n\n"
            "def topo_sort(edges: list[tuple[str, str]]) -> list[str]:\n"
            "    graph: dict[str, list[str]] = defaultdict(list)\n"
            "    indegree: dict[str, int] = {}\n"
            "    for before, after in edges:\n"
            "        graph[before].append(after)\n"
            "        indegree.setdefault(before, 0)\n"
            "        indegree[after] = indegree.get(after, 0) + 1\n"
            "    ready = sorted(node for node, degree in indegree.items() if degree == 0)\n"
            "    order: list[str] = []\n"
            "    while ready:\n"
            "        node = ready.pop(0)\n"
            "        order.append(node)\n"
            "        for child in sorted(graph[node]):\n"
            "            indegree[child] -= 1\n"
            "            if indegree[child] == 0:\n"
            "                ready.append(child)\n"
            "                ready.sort()\n"
            "    if len(order) != len(indegree):\n"
            "        raise ValueError('cycle detected')\n"
            "    return order",
            f"assert topo_sort([('{a}', '{b}'), ('{b}', '{c}')]) == ['{a}', '{b}', '{c}']\n"
            "try:\n"
            "    topo_sort([('a', 'b'), ('b', 'a')])\n"
            "except ValueError as exc:\n"
            "    assert str(exc) == 'cycle detected'\n"
            "else:\n"
            "    raise AssertionError('expected cycle')",
        )

    def binary_search_left(i: int) -> tuple[str, str, str, str]:
        values = sorted({i % 3, i % 3 + 2, i % 3 + 5, i % 3 + 9})
        target = values[2]
        return (
            "binary_search_left",
            f"Implement lower_bound(values, target) without using bisect. Seed case: {i}.",
            "def lower_bound(values: list[int], target: int) -> int:\n"
            "    lo = 0\n"
            "    hi = len(values)\n"
            "    while lo < hi:\n"
            "        mid = (lo + hi) // 2\n"
            "        if values[mid] < target:\n"
            "            lo = mid + 1\n"
            "        else:\n"
            "            hi = mid\n"
            "    return lo",
            f"assert lower_bound({values!r}, {target}) == 2\n"
            f"assert lower_bound({values!r}, {values[0] - 1}) == 0\n"
            f"assert lower_bound({values!r}, {values[-1] + 1}) == {len(values)}",
        )

    def validate_tool_call(i: int) -> tuple[str, str, str, str]:
        required = f"query_{i % 9}"
        return (
            "validate_tool_call",
            f"Implement validate_tool_call(call, required_key) for MCP-style JSON tool calls. Seed case: {i}.",
            "def validate_tool_call(call: dict[str, object], required_key: str) -> bool:\n"
            "    tool_name = call.get('tool_name')\n"
            "    arguments = call.get('arguments')\n"
            "    if not isinstance(tool_name, str) or not tool_name:\n"
            "        return False\n"
            "    if not isinstance(arguments, dict) or not arguments:\n"
            "        return False\n"
            "    return required_key in arguments",
            f"assert validate_tool_call({{'tool_name': 'mcp.search', 'arguments': {{'{required}': 'abc'}}}}, '{required}') is True\n"
            f"assert validate_tool_call({{'tool_name': 'mcp.search', 'arguments': {{}}}}, '{required}') is False\n"
            f"assert validate_tool_call({{'tool_name': '', 'arguments': {{'{required}': 'abc'}}}}, '{required}') is False",
        )

    builders = (
        clamp,
        normalize,
        parse_kv,
        rolling,
        chunk_list,
        safe_ratio,
        merge_intervals,
        topo_sort,
        binary_search_left,
        validate_tool_call,
    )

    def rust_example(i: int) -> tuple[str, str, str, str, str]:
        low = i % 8
        high = low + 10
        value = high + 3
        return (
            "rust2024",
            "rust2024_saturating_clamp",
            f"Implement a Rust 2024 function clamp_i32(value, low, high) -> Result<i32, String>. Seed case: {i}.",
            "pub fn clamp_i32(value: i32, low: i32, high: i32) -> Result<i32, String> {\n"
            "    if low > high {\n"
            "        return Err(\"low must be <= high\".to_string());\n"
            "    }\n"
            "    Ok(value.max(low).min(high))\n"
            "}",
            "# Rust 2024 harness\n"
            "Run with `cargo test --edition 2024`.\n"
            f"assert_eq!(clamp_i32({value}, {low}, {high}).unwrap(), {high});\n"
            f"assert_eq!(clamp_i32({low - 1}, {low}, {high}).unwrap(), {low});\n"
            "assert!(clamp_i32(1, 5, 3).is_err());",
        )

    def go_example(i: int) -> tuple[str, str, str, str, str]:
        key = f"region_{i % 11}"
        value = f"zone_{(i * 7) % 13}"
        return (
            "go",
            "go_parse_label",
            f"Implement Go function ParseLabel(line string) (string, string, error). Seed case: {i}.",
            "package labels\n\n"
            "import (\n"
            "    \"errors\"\n"
            "    \"strings\"\n"
            ")\n\n"
            "func ParseLabel(line string) (string, string, error) {\n"
            "    parts := strings.SplitN(line, \"=\", 2)\n"
            "    if len(parts) != 2 {\n"
            "        return \"\", \"\", errors.New(\"missing separator\")\n"
            "    }\n"
            "    key := strings.TrimSpace(parts[0])\n"
            "    val := strings.TrimSpace(parts[1])\n"
            "    if key == \"\" {\n"
            "        return \"\", \"\", errors.New(\"empty key\")\n"
            "    }\n"
            "    return key, val, nil\n"
            "}",
            "# Go harness\n"
            "Run with `go test ./...`.\n"
            f"k, v, err := ParseLabel(\" {key} = {value} \"); if err != nil || k != \"{key}\" || v != \"{value}\" {{ t.Fatalf(\"unexpected parse\") }}\n"
            "_, _, err = ParseLabel(\"bad\"); if err == nil { t.Fatalf(\"expected error\") }",
        )

    def ts_example(i: int) -> tuple[str, str, str, str, str]:
        limit = 3 + (i % 5)
        return (
            "typescript",
            "typescript_take_unique",
            f"Implement TypeScript function takeUnique(values: readonly string[], limit: number): string[]. Seed case: {i}.",
            "export function takeUnique(values: readonly string[], limit: number): string[] {\n"
            "  if (!Number.isInteger(limit) || limit < 0) {\n"
            "    throw new Error(\"limit must be a non-negative integer\");\n"
            "  }\n"
            "  const seen = new Set<string>();\n"
            "  const out: string[] = [];\n"
            "  for (const raw of values) {\n"
            "    const value = raw.trim().toLowerCase();\n"
            "    if (value && !seen.has(value)) {\n"
            "      seen.add(value);\n"
            "      out.push(value);\n"
            "      if (out.length === limit) break;\n"
            "    }\n"
            "  }\n"
            "  return out;\n"
            "}",
            "# TypeScript harness\n"
            "Run with `npm test` or `tsc --noEmit --strict` plus a test runner.\n"
            f"expect(takeUnique([\" A \", \"a\", \"B\", \"C\"], {limit})).toEqual([\"a\", \"b\", \"c\"].slice(0, {limit}));\n"
            "expect(() => takeUnique([\"a\"], -1)).toThrow(\"limit must be a non-negative integer\");",
        )

    def csharp_example(i: int) -> tuple[str, str, str, str, str]:
        step = 2 + (i % 4)
        expected = sum([1, 2, 3, 4, 5, 6, 7, 8, 9][idx] for idx in range(0, 9, step))
        return (
            "csharp",
            "csharp_stride_sum",
            f"Implement C# static method SumEveryNth(IReadOnlyList<int> values, int step). Seed case: {i}.",
            "using System;\n"
            "using System.Collections.Generic;\n\n"
            "public static class SeriesTools\n"
            "{\n"
            "    public static int SumEveryNth(IReadOnlyList<int> values, int step)\n"
            "    {\n"
            "        if (step <= 0)\n"
            "        {\n"
            "            throw new ArgumentOutOfRangeException(nameof(step), \"step must be positive\");\n"
            "        }\n"
            "        var total = 0;\n"
            "        for (var index = 0; index < values.Count; index += step)\n"
            "        {\n"
            "            total += values[index];\n"
            "        }\n"
            "        return total;\n"
            "    }\n"
            "}",
            "# C# harness\n"
            "Run with `dotnet test`.\n"
            f"Assert.Equal({expected}, SeriesTools.SumEveryNth(new[] {{1, 2, 3, 4, 5, 6, 7, 8, 9}}, {step}));\n"
            "Assert.Throws<ArgumentOutOfRangeException>(() => SeriesTools.SumEveryNth(new[] {1}, 0));",
        )

    static_builders = (rust_example, go_example, ts_example, csharp_example)
    for i in range(count):
        language_slot = i % 5
        if language_slot == 0:
            name, request, code, verifier = builders[(i // 5) % len(builders)](i)
            language = "python"
            target_kind = "python_exec"
        else:
            language, name, request, code, verifier = static_builders[language_slot - 1](i)
            target_kind = "code_static_spec"
        task = _task("code", name, request, target_kind, i)
        yield SyntheticExample(
            task=task,
            example={
                "user_request": request,
                "assistant_code": code,
                "verifier_snippet": verifier,
                "language": language,
                "rationale": f"compact verifier-backed {language} code seed",
            },
        )


def _math_examples(count: int) -> Iterable[SyntheticExample]:
    def linear(i: int) -> tuple[str, str, str, str]:
        a = 2 + (i % 7)
        x = 3 + (i % 11)
        b = 5 + (i % 13)
        c = a * x + b
        return (
            "linear_equation",
            f"Solve the linear equation {a}x + {b} = {c} and report the exact value of x. Synthetic seed id {i}.",
            f"Subtract {b} from both sides to get {a}x = {c - b}. Dividing by {a} gives x = {x}.",
            str(x),
        )

    def rectangle(i: int) -> tuple[str, str, str, str]:
        width = 4 + (i % 9)
        height = 5 + ((i * 2) % 8)
        divisor = 2 + (i % 4)
        area = width * height
        answer = area // divisor if area % divisor == 0 else f"{area}/{divisor}"
        return (
            "area_fraction",
            f"A rectangle has width {width} and height {height}. Compute exactly one {divisor}th of its area. Synthetic seed id {i}.",
            f"The area is {width} times {height} = {area}. One {divisor}th of the area is {area}/{divisor}, which is {answer}.",
            str(answer),
        )

    def probability(i: int) -> tuple[str, str, str, str]:
        sides = 6 + (i % 5)
        threshold = 2 + (i % (sides - 2))
        favorable = sides - threshold + 1
        return (
            "die_probability",
            f"A fair {sides}-sided die is rolled once. What is P(result is at least {threshold})? Synthetic seed id {i}.",
            f"The favorable outcomes are {threshold} through {sides}, so there are {favorable} favorable outcomes out of {sides}. The probability is {favorable}/{sides}.",
            f"{favorable}/{sides}",
        )

    def arithmetic_sum(i: int) -> tuple[str, str, str, str]:
        n = 6 + (i % 15)
        step = 2 + (i % 5)
        answer = step * n * (n + 1) // 2
        return (
            "arithmetic_series",
            f"Find the exact sum of the first {n} positive multiples of {step} using a finite series formula. Synthetic seed id {i}.",
            f"The terms are {step} times 1 through {n}. The sum is {step} * {n} * ({n}+1) / 2 = {answer}.",
            str(answer),
        )

    def weighted_average(i: int) -> tuple[str, str, str, str]:
        a = 1 + (i % 8)
        b = 2 + ((i * 3) % 7)
        wa = 2 + (i % 4)
        wb = 3 + ((i * 2) % 5)
        numerator = a * wa + b * wb
        denominator = wa + wb
        answer = numerator // denominator if numerator % denominator == 0 else f"{numerator}/{denominator}"
        return (
            "weighted_average",
            f"Compute the exact weighted average of values {a} and {b} with weights {wa} and {wb}. Synthetic seed id {i}.",
            f"The weighted sum is {a}*{wa} + {b}*{wb} = {numerator}. The total weight is {denominator}. The exact average is {answer}.",
            str(answer),
        )

    def modular_remainder(i: int) -> tuple[str, str, str, str]:
        a = 7 + (i % 13)
        b = 3 + ((i * 5) % 17)
        modulus = 5 + (i % 9)
        answer = (a * b + b) % modulus
        return (
            "modular_arithmetic",
            f"Compute the remainder of ({a}*{b}+{b}) modulo {modulus}. Synthetic seed id {i}.",
            f"First compute {a}*{b}+{b} = {a * b + b}. Dividing by {modulus} leaves remainder {answer}.",
            str(answer),
        )

    def bayes_binary(i: int) -> tuple[str, str, str, str]:
        prevalence_num = 1 + (i % 4)
        sensitivity_num = 7 + (i % 3)
        false_positive_num = 1 + (i % 2)
        prevalence = prevalence_num / 10
        sensitivity = sensitivity_num / 10
        false_positive = false_positive_num / 10
        numerator = prevalence_num * sensitivity_num
        denominator = numerator + (10 - prevalence_num) * false_positive_num
        return (
            "bayes_binary",
            "A test has prevalence "
            f"{prevalence:.1f}, sensitivity {sensitivity:.1f}, and false positive rate {false_positive:.1f}. "
            f"Compute P(disease | positive) exactly as a fraction. Synthetic seed id {i}.",
            "Bayes rule gives sensitivity*prevalence divided by "
            "sensitivity*prevalence + false_positive_rate*(1-prevalence). "
            f"Using tenths, the numerator is {numerator} and the denominator is {denominator}.",
            f"{numerator}/{denominator}",
        )

    def integer_quadratic_root(i: int) -> tuple[str, str, str, str]:
        r1 = 2 + (i % 8)
        r2 = r1 + 3 + (i % 5)
        s = r1 + r2
        p = r1 * r2
        return (
            "quadratic_roots",
            f"The equation x^2 - {s}x + {p} = 0 has two positive integer roots. Report the smaller root. Synthetic seed id {i}.",
            f"The factorization is (x - {r1})(x - {r2}) = 0, so the roots are {r1} and {r2}. The smaller root is {r1}.",
            str(r1),
        )

    def vector_dot(i: int) -> tuple[str, str, str, str]:
        a = [1 + (i % 4), 2 + (i % 5), 3 + (i % 6)]
        b = [2 + (i % 3), 1 + ((i * 2) % 4), 4 + ((i * 3) % 5)]
        answer = sum(x * y for x, y in zip(a, b))
        return (
            "vector_dot_product",
            f"Compute the dot product of vectors {a} and {b}. Synthetic seed id {i}.",
            f"Multiply componentwise and add: {a[0]}*{b[0]} + {a[1]}*{b[1]} + {a[2]}*{b[2]} = {answer}.",
            str(answer),
        )

    def matrix_determinant(i: int) -> tuple[str, str, str, str]:
        a = 1 + (i % 5)
        b = 2 + (i % 7)
        c = 3 + ((i * 2) % 5)
        d = 4 + ((i * 3) % 7)
        answer = a * d - b * c
        return (
            "matrix_determinant",
            f"Compute the determinant of the 2 by 2 matrix [[{a}, {b}], [{c}, {d}]]. Synthetic seed id {i}.",
            f"For a 2 by 2 matrix, det = ad - bc = {a}*{d} - {b}*{c} = {answer}.",
            str(answer),
        )

    def polynomial_derivative(i: int) -> tuple[str, str, str, str]:
        a = 2 + (i % 6)
        b = 3 + ((i * 2) % 7)
        c = 5 + ((i * 3) % 8)
        x = 1 + (i % 5)
        answer = 3 * a * x * x + 2 * b * x + c
        return (
            "polynomial_derivative",
            f"Let f(x) = {a}x^3 + {b}x^2 + {c}x + 7. Compute f'({x}). Synthetic seed id {i}.",
            f"The derivative is f'(x) = {3*a}x^2 + {2*b}x + {c}. Evaluating at {x} gives {answer}.",
            str(answer),
        )

    def definite_integral(i: int) -> tuple[str, str, str, str]:
        a = 1 + (i % 5)
        b = 2 + ((i * 2) % 5)
        n = 2 + (i % 4)
        numerator = a * n ** 3 + 3 * b * n
        return (
            "definite_integral",
            f"Compute the exact value of integral from 0 to {n} of ({3*a}x^2 + {b}) dx. Synthetic seed id {i}.",
            f"An antiderivative is {a}x^3 + {b}x. Evaluating from 0 to {n} gives {a}*{n}^3 + {b}*{n} = {numerator}.",
            str(numerator),
        )

    def conditional_probability(i: int) -> tuple[str, str, str, str]:
        total = 40 + (i % 20)
        a_count = 10 + (i % 8)
        b_count = 12 + ((i * 3) % 9)
        both = 3 + (i % min(a_count, b_count, 7))
        return (
            "conditional_probability",
            f"In a cohort of {total}, event A occurs in {a_count}, event B occurs in {b_count}, and both occur in {both}. Compute P(A | B). Synthetic seed id {i}.",
            f"Conditional probability P(A|B) is count(A and B) divided by count(B), so the exact value is {both}/{b_count}.",
            f"{both}/{b_count}",
        )

    def geometric_recurrence(i: int) -> tuple[str, str, str, str]:
        first = 2 + (i % 5)
        ratio = 2 + ((i * 3) % 4)
        n = 4 + (i % 5)
        answer = first * ratio ** (n - 1)
        return (
            "geometric_recurrence",
            f"A recurrence has a_1 = {first} and a_n = {ratio} a_(n-1). Compute a_{n}. Synthetic seed id {i}.",
            f"This is a geometric sequence, so a_{n} = {first} * {ratio}^({n}-1) = {answer}.",
            str(answer),
        )

    def inclusion_exclusion(i: int) -> tuple[str, str, str, str]:
        a_count = 15 + (i % 10)
        b_count = 14 + ((i * 2) % 9)
        both = 4 + (i % 6)
        answer = a_count + b_count - both
        return (
            "inclusion_exclusion",
            f"A finite set has |A|={a_count}, |B|={b_count}, and |A intersection B|={both}. Compute |A union B|. Synthetic seed id {i}.",
            f"By inclusion-exclusion, |A union B| = |A| + |B| - |A intersection B| = {a_count}+{b_count}-{both} = {answer}.",
            str(answer),
        )

    builders = (
        linear,
        rectangle,
        probability,
        arithmetic_sum,
        weighted_average,
        modular_remainder,
        bayes_binary,
        integer_quadratic_root,
        vector_dot,
        matrix_determinant,
        polynomial_derivative,
        definite_integral,
        conditional_probability,
        geometric_recurrence,
        inclusion_exclusion,
    )
    for i in range(count):
        name, question, reasoning, answer = builders[i % len(builders)](i)
        task = _task("math", name, question, "exact_math", i)
        yield SyntheticExample(
            task=task,
            example={
                "question": question,
                "reasoning": reasoning,
                "final_answer": answer,
                "reference": answer,
            },
        )


def _stem_examples(count: int) -> Iterable[SyntheticExample]:
    specs = [
        (
            "physics_energy",
            "A cart's speed doubles while mass stays fixed in a low-friction track experiment. How does kinetic energy change?",
            "It quadruples because kinetic energy scales with the square of speed",
            [
                "It is unchanged because only direction matters",
                "It doubles because speed appears linearly",
                "It halves because energy is inversely related to speed",
            ],
            "Kinetic energy is proportional to v^2, so doubling speed multiplies energy by four.",
        ),
        (
            "chem_buffer",
            "Which mixture most directly resists pH change when a small amount of acid is added to an aqueous sample?",
            "Weak acid plus conjugate base in comparable amounts",
            [
                "Pure water without a conjugate acid-base pair",
                "Strong acid alone with no buffering partner",
                "Neutral salt only without weak acid chemistry",
            ],
            "A buffer needs a weak acid/base pair and its conjugate partner to consume added acid or base.",
        ),
        (
            "medicine_screening",
            "For a rare disease screening test with fixed sensitivity and specificity, what usually happens to positive predictive value when prevalence rises?",
            "It increases because true positives become a larger share of positive tests",
            [
                "It decreases because sensitivity becomes lower",
                "It becomes zero because false positives dominate all positives",
                "It is unrelated because prevalence never affects predictive value",
            ],
            "With sensitivity and specificity fixed, higher prevalence increases the fraction of positives that are true positives.",
        ),
        (
            "statistics_ci",
            "If sample size increases fourfold while the population variance is fixed, how does the standard error of the sample mean change?",
            "It halves because standard error is inversely proportional to square root of sample size",
            [
                "It doubles because sample size appears in the numerator",
                "It quadruples because sample size was multiplied by four",
                "It is unchanged because variance was held fixed",
            ],
            "Standard error scales as 1/sqrt(n), so multiplying n by four divides it by two.",
        ),
        (
            "biology_enzyme",
            "In a controlled enzyme-catalyzed reaction, which change most directly lowers the initial reaction rate?",
            "Adding a competitive inhibitor that occupies active sites",
            [
                "Doubling substrate concentration when the enzyme is already saturated",
                "Maintaining the enzyme at its optimal pH and temperature",
                "Adding more active enzyme molecules to the same reaction volume",
            ],
            "A competitive inhibitor reduces active-site availability, lowering initial rate under otherwise comparable conditions.",
        ),
        (
            "electrical_power",
            "A resistor's voltage is doubled while resistance is fixed. How does electrical power dissipated in the resistor change?",
            "It quadruples because P = V^2/R when resistance is fixed",
            [
                "It is unchanged because resistance did not change",
                "It doubles because voltage appears linearly in all power laws",
                "It halves because current must decrease",
            ],
            "For a fixed resistor, P = V^2/R, so doubling V multiplies power by four.",
        ),
        (
            "medical_risk",
            "A medication reduces relative risk by 25 percent from a baseline event risk of 20 percent. What is the absolute risk reduction?",
            "Five percentage points because 25 percent of 20 percent is 5 percent",
            [
                "Twenty-five percentage points because relative and absolute risk are identical",
                "Twenty percentage points because the baseline risk is twenty percent",
                "Zero percentage points because relative risk does not affect absolute risk",
            ],
            "Absolute risk reduction is baseline risk times relative reduction: 0.20 * 0.25 = 0.05.",
        ),
        (
            "materials_stress",
            "A metal rod is loaded within its elastic region. Which quantity is the slope of the stress-strain curve?",
            "Young's modulus because it relates stress to strain in the linear elastic region",
            [
                "Density because mass per volume sets the slope of all mechanical curves",
                "Thermal conductivity because heat transfer controls elastic slope",
                "Specific heat because stored heat determines strain",
            ],
            "In the elastic linear regime, stress equals Young's modulus times strain.",
        ),
        (
            "epidemiology_confounding",
            "A study finds coffee drinkers have higher disease risk, but coffee drinkers also smoke more often. What is smoking in this setup?",
            "A potential confounder because it is associated with exposure and outcome",
            [
                "A mediator that must always lie after coffee in the causal chain",
                "A randomized treatment because it balances groups automatically",
                "A negative control outcome because it cannot affect disease",
            ],
            "A confounder is related to both the exposure and the outcome and can bias the association.",
        ),
        (
            "astronomy_inverse_square",
            "A star-like point source is observed from twice the distance. How does measured flux change in empty space?",
            "It becomes one quarter because flux follows an inverse-square law",
            [
                "It doubles because the line of sight is longer",
                "It is unchanged because luminosity is constant",
                "It becomes half because distance changed by a factor of two",
            ],
            "For a fixed luminosity point source, flux scales as 1/r^2.",
        ),
        (
            "linear_algebra_rank",
            "Two rows of a 2 by 2 matrix are exact multiples of each other. What can be concluded about its determinant?",
            "The determinant is zero because the rows are linearly dependent",
            [
                "The determinant must be one because the rows are proportional",
                "The determinant is negative because one row repeats information",
                "No conclusion is possible because determinants ignore rows",
            ],
            "A square matrix with linearly dependent rows has zero determinant.",
        ),
        (
            "clinical_trial_power",
            "All else equal, which design change usually increases statistical power in a randomized trial?",
            "Increasing sample size because it reduces standard error",
            [
                "Using a smaller true effect while keeping noise fixed",
                "Increasing measurement noise without changing sample size",
                "Removing randomization and comparing arbitrary groups",
            ],
            "Larger sample size usually lowers uncertainty and increases the chance of detecting a true effect.",
        ),
        (
            "pharmacokinetics_half_life",
            "For a first-order drug elimination process, what happens to plasma concentration after one half-life?",
            "It falls to one half because first-order half-life is the time for 50 percent reduction",
            [
                "It becomes zero because all drug is eliminated after one half-life",
                "It doubles because elimination accelerates concentration",
                "It remains unchanged because half-life only applies to radioactive atoms",
            ],
            "First-order elimination produces a constant fractional decrease per half-life.",
        ),
        (
            "control_feedback",
            "In a stable negative-feedback control loop, what is the primary effect of increasing proportional gain too far?",
            "It can increase overshoot or oscillation because the controller reacts too aggressively",
            [
                "It always removes overshoot because larger gain is unconditionally stabilizing",
                "It disables feedback and makes the system open-loop",
                "It changes the sensor units but cannot affect dynamics",
            ],
            "High proportional gain can reduce steady-state error but may reduce damping and cause overshoot.",
        ),
        (
            "signal_aliasing",
            "A sinusoid is sampled below twice its frequency. Which failure mode is expected?",
            "Aliasing because the sampling rate violates the Nyquist criterion",
            [
                "Perfect reconstruction because all sinusoids need only one sample",
                "Increased bit depth because sampling rate changes amplitude quantization",
                "Thermal noise cancellation because slower sampling averages noise exactly",
            ],
            "The Nyquist criterion requires sampling at more than twice the highest frequency to avoid aliasing.",
        ),
        (
            "genetics_autosomal_recessive",
            "Two heterozygous carriers for an autosomal recessive condition have a child. What is the probability the child is affected?",
            "One quarter because only the homozygous recessive genotype is affected",
            [
                "One half because each parent has one recessive allele",
                "Three quarters because most children inherit at least one variant",
                "Zero because carriers cannot have affected children",
            ],
            "A carrier-by-carrier cross gives genotypes AA, Aa, Aa, aa, so affected probability is 1/4.",
        ),
        (
            "renal_clearance",
            "If a substance is freely filtered and neither secreted nor reabsorbed, what does its renal clearance approximate?",
            "Glomerular filtration rate because excretion equals filtered load",
            [
                "Renal plasma flow because every plasma molecule is cleared",
                "Zero because filtered substances are never excreted",
                "Tubular maximum because secretion is saturated",
            ],
            "For a freely filtered substance with no reabsorption or secretion, clearance tracks GFR.",
        ),
        (
            "machine_learning_calibration",
            "A classifier assigns probability 0.8 to many cases, but only 60 percent are actually positive. What issue is present?",
            "Miscalibration because predicted probabilities overstate observed frequencies",
            [
                "Perfect calibration because 0.8 is greater than 0.6",
                "Class imbalance only, which cannot affect probability estimates",
                "Data leakage, which is proven by any nonzero error",
            ],
            "Calibration compares predicted probabilities with empirical outcome frequencies.",
        ),
        (
            "thermodynamics_adiabatic",
            "In an adiabatic compression of an ideal gas with no heat exchange, what generally happens to temperature?",
            "It increases because work done on the gas raises internal energy",
            [
                "It must decrease because volume decreases",
                "It stays fixed because adiabatic means isothermal",
                "It becomes zero because heat transfer is zero",
            ],
            "With no heat exchange, compression work increases internal energy and therefore temperature for an ideal gas.",
        ),
        (
            "medical_test_likelihood_ratio",
            "A diagnostic test has a positive likelihood ratio greater than 1. What does a positive result do to disease odds?",
            "It increases post-test odds because LR+ multiplies pre-test odds by a factor above one",
            [
                "It decreases odds because all tests introduce false positives",
                "It leaves odds unchanged because likelihood ratios do not affect Bayes updates",
                "It proves disease with probability exactly one",
            ],
            "Bayesian updating multiplies prior odds by the likelihood ratio.",
        ),
    ]
    for i in range(count):
        name, question, correct_choice, distractors, rationale = specs[i % len(specs)]
        question = f"{question} Synthetic seed id {i}."
        expected = "ABCD"[i % 4]
        choices = list(distractors[:3])
        choices.insert("ABCD".index(expected), correct_choice)
        answer = expected
        task = _task("stem_reasoning", name, question, "mcq_reasoning", i)
        yield SyntheticExample(
            task=task,
            example={
                "question": question,
                "choices": choices,
                "answer": answer,
                "final_choice": answer,
                "reasoning": rationale,
                "rationale": rationale,
                "reference": answer,
            },
        )


def _tool_examples(count: int) -> Iterable[SyntheticExample]:
    specs = [
        (
            "mcp_file_search",
            "Search project docs for checkpoint retention rules.",
            "mcp.files.search",
            {"query": "rolling checkpoint keep 3", "root": "H:/elt_data", "read_only": True},
        ),
        (
            "agent_unit_tests",
            "Run focused unit tests for the synthetic dataset builder.",
            "agent.test.run",
            {"command": "uv run --no-sync pytest -q tests/test_synthetic_v1_seed.py", "timeout_sec": 120, "read_only": True},
        ),
        (
            "mcp_dataset_card",
            "Create a dataset card draft for a verifier-backed SFT bundle.",
            "mcp.docs.write",
            {"path": "H:/elt_data/synthetic_v1_seed/README.md", "title": "Synthetic v1 seed", "dry_run": True},
        ),
        (
            "agent_benchmark_plan",
            "Plan an anytime L=1..4 evaluation sweep for code and math lanes.",
            "agent.eval.plan",
            {"lanes": ["code", "math"], "loops": [1, 2, 3, 4], "read_only": True},
        ),
        (
            "mcp_issue_triage",
            "Triage open quality-gate failures without modifying files.",
            "mcp.github.issue_search",
            {"query": "label:quality-gate verifier", "repo": "zapabob/elastic-looped-transformer", "read_only": True},
        ),
        (
            "agent_patch_plan",
            "Draft a bounded patch plan for failing math verifier cases.",
            "agent.plan.create",
            {"scope": "math verifier dry-run", "max_steps": 5, "dry_run": True},
        ),
        (
            "mcp_metric_read",
            "Read latest pipeline progress metrics from H drive.",
            "mcp.metrics.read",
            {"path": "H:/elt_data/pipeline_state/progress_report.json", "read_only": True},
        ),
        (
            "mcp_agent_harness",
            "Construct a read-only MCP harness invocation for a local code agent.",
            "mcp.agent.invoke",
            {"agent": "code-reviewer", "cwd": "C:/Users/downl/Desktop/新しいフォルダー (7)", "mode": "read_only"},
        ),
        (
            "mcp_schema_validate",
            "Validate a tool-call JSON schema before executing an agent plan.",
            "mcp.schema.validate",
            {"schema_name": "tool_call_v1", "strict": True, "dry_run": True},
        ),
        (
            "agent_static_analysis",
            "Run static analysis in no-write mode for a Python training utility.",
            "agent.static_analysis.run",
            {"targets": ["src/elt_lm/synthetic_v1_seed.py"], "write": False, "timeout_sec": 180},
        ),
        (
            "mcp_checkpoint_status",
            "Read rolling checkpoint freshness and keep count for the active run.",
            "mcp.training.checkpoints",
            {"run_dir": "H:/elt_data/runs/base_1B_clean_replay_phase2", "keep": 3, "read_only": True},
        ),
        (
            "agent_tool_router",
            "Route a user request to a safe read-only tool without executing mutations.",
            "agent.tool.route",
            {"request_type": "dataset_audit", "allowed_modes": ["read_only", "dry_run"], "dry_run": True},
        ),
        (
            "mcp_tool_discovery",
            "Discover available MCP tools for a coding-agent task without invoking them.",
            "mcp.tools.list",
            {"server": "codex_apps", "capability": "code_quality", "read_only": True},
        ),
        (
            "mcp_resource_read",
            "Read a specific MCP resource that describes benchmark schema.",
            "mcp.resources.read",
            {"server": "elt-local", "uri": "resource://benchmarks/schema/v1", "read_only": True},
        ),
        (
            "agent_plan_execute_dry_run",
            "Run an implementation plan through an agent harness in dry-run mode.",
            "agent.plan.execute",
            {"plan_id": "synthetic-plan", "max_steps": 6, "dry_run": True, "write": False},
        ),
        (
            "agent_memory_search",
            "Search project memory for prior LoRA checkpoint decisions.",
            "agent.memory.search",
            {"query": "side lora adapter checkpoint synthetic gb", "limit": 5, "read_only": True},
        ),
        (
            "agent_browser_inspect",
            "Inspect a local dashboard route without clicking or mutating state.",
            "agent.browser.inspect",
            {"url": "http://localhost:7860/status", "selectors": ["#stage", "#loss"], "read_only": True},
        ),
        (
            "agent_security_scan",
            "Run a non-mutating security scan for generated tool-call examples.",
            "agent.security.scan",
            {"targets": ["H:/elt_data/synthetic_v1_tool_gb"], "severity": "high", "dry_run": True},
        ),
        (
            "mcp_training_resume",
            "Query whether a training run can resume from the latest rolling checkpoint.",
            "mcp.training.resume_status",
            {"run_dir": "H:/elt_data/runs/qwen35_4b_side_lora_tool_sft_synthetic_gb", "read_only": True},
        ),
        (
            "agent_eval_rerank",
            "Plan a verifier rerank pass for tool-use predictions without executing tools.",
            "agent.eval.rerank",
            {"lane": "tool_use", "verifiers": ["json_match"], "top_k": 4, "dry_run": True},
        ),
        (
            "mcp_dataset_sample",
            "Sample a dataset shard for manual review using read-only access.",
            "mcp.dataset.sample",
            {"path": "H:/elt_data/synthetic_v1_tool_gb/tool_use/distill_train.jsonl", "sample_size": 8, "read_only": True},
        ),
        (
            "agent_ci_matrix",
            "Construct a CI matrix for Rust, Go, TypeScript, Python, and C# code tasks.",
            "agent.ci.matrix",
            {"languages": ["rust2024", "go", "typescript", "python", "csharp"], "dry_run": True},
        ),
    ]
    for i in range(count):
        name, request, tool_name, arguments = specs[i % len(specs)]
        request = f"{request} Synthetic seed id {i}."
        arguments = dict(arguments)
        arguments["request_id"] = f"synthetic-{i}"
        task = _task("tool_use", name, request, "json_match", i)
        yield SyntheticExample(
            task=task,
            example={
                "user_request": request,
                "tool_name": tool_name,
                "arguments": arguments,
                "reference": {"tool_name": tool_name, "arguments": arguments},
            },
        )


def generate_lane_examples(lane: str, count: int) -> list[SyntheticExample]:
    if lane == "code":
        return list(_code_examples(count))
    if lane == "math":
        return list(_math_examples(count))
    if lane == "stem_reasoning":
        return list(_stem_examples(count))
    if lane == "tool_use":
        return list(_tool_examples(count))
    raise ValueError(f"unsupported lane: {lane}")


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _record_fingerprint(value: str) -> str:
    return hashlib.sha1(" ".join(value.strip().lower().split()).encode("utf-8")).hexdigest()


def _lane_iterator(lane: str) -> Iterable[SyntheticExample]:
    if lane == "code":
        return _code_examples(2_147_483_647)
    if lane == "math":
        return _math_examples(2_147_483_647)
    if lane == "stem_reasoning":
        return _stem_examples(2_147_483_647)
    if lane == "tool_use":
        return _tool_examples(2_147_483_647)
    raise ValueError(f"unsupported lane: {lane}")


def _empty_stream_summary(lane: str) -> dict[str, Any]:
    return {
        "lane": lane,
        "source": "synthetic-v1-seed",
        "total_records": 0,
        "valid_json_records": 0,
        "schema_valid_rate": 0.0,
        "unique_text_ratio": 0.0,
        "exact_duplicate_count": 0,
        "exact_duplicate_ratio": 0.0,
        "duplicate_prompt_count": 0,
        "fallback_reject_count": 0,
        "verifier_pass_rate": 0.0,
        "verifier_pass_count": 0,
        "verifier_total": 0,
        "sample_verifier_pass_rate": 0.0,
        "sample_verifier_pass_count": 0,
        "sample_verifier_total": 0,
        "answer_distribution": {},
        "accepted_records": 0,
        "attempted_tasks": 0,
        "generation_attempts": 0,
        "quality_reject_count": 0,
        "quality_reject_reasons": {},
        "domain_counts": {},
        "label_counts": {},
        "split_counts": {},
        "lane_counts": {},
        "task_counts": {},
        "train_bytes": 0,
        "val_bytes": 0,
        "total_bytes": 0,
        "rejected": {},
    }


def build_synthetic_seed_bundle_to_target(
    *,
    output_root: Path,
    target_bytes: int,
    val_ratio: float,
    lanes: Iterable[str] = LANES,
    validation_sample_per_lane: int = 256,
) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    lanes_list = list(lanes)
    if target_bytes <= 0:
        raise ValueError("target_bytes must be positive")
    target_per_lane = max(1, target_bytes // max(1, len(lanes_list)))
    overall: dict[str, Any] = {
        "output_root": str(output_root),
        "target_bytes": target_bytes,
        "target_per_lane": target_per_lane,
        "validation_sample_per_lane": validation_sample_per_lane,
        "lanes": {},
    }
    split_mod = max(2, round(1.0 / max(val_ratio, 1e-6)))

    for lane in lanes_list:
        lane_dir = output_root / lane
        lane_dir.mkdir(parents=True, exist_ok=True)
        train_path = lane_dir / "distill_train.jsonl"
        val_path = lane_dir / "distill_val.jsonl"
        summary = _empty_stream_summary(lane)
        domain_counts: Counter[str] = Counter()
        label_counts: Counter[str] = Counter()
        split_counts: Counter[str] = Counter()
        task_counts: Counter[str] = Counter()
        answer_distribution: Counter[str] = Counter()
        reject_counts: Counter[str] = Counter()
        seen_text: set[str] = set()
        seen_prompt: set[str] = set()
        sample_verified = 0
        sample_pass = 0

        with train_path.open("w", encoding="utf-8") as train_f, val_path.open("w", encoding="utf-8") as val_f:
            for idx, item in enumerate(_lane_iterator(lane)):
                if summary["total_bytes"] >= target_per_lane:
                    break
                split = "val" if (idx % split_mod == 0) else "train"
                record = build_sft_record(
                    task=item.task,
                    example=item.example,
                    teacher_name="synthetic-v1-seed",
                    split=split,
                )
                text_fp = _record_fingerprint(str(record["text"]))
                prompt_fp = _record_fingerprint(str(record["prompt"]))
                summary["generation_attempts"] += 1
                summary["attempted_tasks"] += 1
                if text_fp in seen_text:
                    reject_counts["duplicate_text"] += 1
                    continue
                if prompt_fp in seen_prompt:
                    reject_counts["duplicate_prompt"] += 1
                    continue

                if sample_verified < validation_sample_per_lane:
                    try:
                        validate_distill_record_quality(record, item.example, item.task, None)
                    except DistillQualityError as exc:
                        reject_counts[str(exc)] += 1
                        continue
                    sample_verified += 1
                    sample_pass += 1

                line = json.dumps(record, ensure_ascii=False) + "\n"
                encoded_len = len(line.encode("utf-8"))
                if split == "val":
                    val_f.write(line)
                    summary["val_bytes"] += encoded_len
                else:
                    train_f.write(line)
                    summary["train_bytes"] += encoded_len
                summary["total_bytes"] += encoded_len
                summary["total_records"] += 1
                summary["valid_json_records"] += 1
                summary["accepted_records"] += 1
                seen_text.add(text_fp)
                seen_prompt.add(prompt_fp)
                metadata = record.get("metadata") or {}
                task_name = str(metadata.get("task_name", "unknown"))
                task_kind = str(record.get("task", "unknown"))
                domain_counts[task_name] += 1
                task_counts[task_name] += 1
                label_counts[task_kind] += 1
                split_counts[split] += 1
                if lane == "stem_reasoning":
                    answer_distribution[str(record.get("reference", "")).strip().upper()] += 1

        total = int(summary["total_records"])
        summary["schema_valid_rate"] = 1.0 if total else 0.0
        summary["unique_text_ratio"] = 1.0 if total else 0.0
        summary["exact_duplicate_ratio"] = 0.0
        summary["verifier_total"] = sample_verified
        summary["verifier_pass_count"] = sample_pass
        summary["verifier_pass_rate"] = sample_pass / sample_verified if sample_verified else 0.0
        summary["sample_verifier_total"] = sample_verified
        summary["sample_verifier_pass_count"] = sample_pass
        summary["sample_verifier_pass_rate"] = summary["verifier_pass_rate"]
        summary["quality_reject_count"] = sum(reject_counts.values())
        summary["quality_reject_reasons"] = dict(reject_counts)
        summary["rejected"] = dict(reject_counts)
        summary["domain_counts"] = dict(domain_counts)
        summary["label_counts"] = dict(label_counts)
        summary["split_counts"] = dict(split_counts)
        summary["lane_counts"] = {lane: total}
        summary["task_counts"] = dict(task_counts)
        summary["answer_distribution"] = dict(answer_distribution)
        (lane_dir / "eval_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        (lane_dir / "README.md").write_text(
            f"# Synthetic v1 large: {lane}\n\n"
            "Large verifier-backed synthetic data generated without teacher sampling.\n\n"
            f"- Records: {summary['total_records']}\n"
            f"- Total bytes: {summary['total_bytes']}\n"
            f"- Sample verifier pass rate: {summary.get('sample_verifier_pass_rate', 0.0):.3f}\n"
            f"- Unique text ratio: {summary.get('unique_text_ratio', 0.0):.3f}\n",
            encoding="utf-8",
        )
        overall["lanes"][lane] = summary
    overall["total_bytes"] = sum(int(item.get("total_bytes", 0)) for item in overall["lanes"].values())
    overall["total_records"] = sum(int(item.get("total_records", 0)) for item in overall["lanes"].values())
    (output_root / "summary.json").write_text(json.dumps(overall, ensure_ascii=False, indent=2), encoding="utf-8")
    return overall


def build_synthetic_seed_bundle(
    *,
    output_root: Path,
    records_per_lane: int,
    val_ratio: float,
    lanes: Iterable[str] = LANES,
) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    overall: dict[str, Any] = {"output_root": str(output_root), "lanes": {}}
    for lane in lanes:
        examples = generate_lane_examples(lane, records_per_lane)
        seen_text: set[str] = set()
        seen_prompt: set[str] = set()
        train_records: list[dict[str, Any]] = []
        val_records: list[dict[str, Any]] = []
        reject_counts: Counter[str] = Counter()
        for idx, item in enumerate(examples):
            split = "val" if (idx % max(2, round(1.0 / max(val_ratio, 1e-6))) == 0) else "train"
            record = build_sft_record(task=item.task, example=item.example, teacher_name="synthetic-v1-seed", split=split)
            try:
                validate_distill_record_quality(
                    record,
                    item.example,
                    item.task,
                    None,
                    seen_text_fingerprints=seen_text,
                    seen_prompt_fingerprints=seen_prompt,
                )
            except DistillQualityError as exc:
                reject_counts[str(exc)] += 1
                continue
            seen_text.add(" ".join(str(record["text"]).strip().lower().split()))
            seen_prompt.add(" ".join(str(record["prompt"]).strip().lower().split()))
            (val_records if split == "val" else train_records).append(record)
        lane_dir = output_root / lane
        all_records = [*train_records, *val_records]
        _write_jsonl(lane_dir / "distill_train.jsonl", train_records)
        _write_jsonl(lane_dir / "distill_val.jsonl", val_records)
        summary = evaluate_distill_records(all_records, quality_counters=reject_counts, run_verifiers=True)
        summary["lane"] = lane
        summary["source"] = "synthetic-v1-seed"
        summary["rejected"] = dict(reject_counts)
        (lane_dir / "eval_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        (lane_dir / "README.md").write_text(
            f"# Synthetic v1 seed: {lane}\n\n"
            "Verifier-backed synthetic seed data generated without teacher sampling.\n\n"
            f"- Records: {summary['total_records']}\n"
            f"- Verifier pass rate: {summary.get('verifier_pass_rate', 0.0):.3f}\n"
            f"- Unique text ratio: {summary.get('unique_text_ratio', 0.0):.3f}\n",
            encoding="utf-8",
        )
        overall["lanes"][lane] = summary
    (output_root / "summary.json").write_text(json.dumps(overall, ensure_ascii=False, indent=2), encoding="utf-8")
    return overall


def cli() -> None:
    parser = argparse.ArgumentParser(description="Build verifier-backed synthetic v1 seed SFT bundles.")
    parser.add_argument("--output-root", type=Path, default=Path("H:/elt_data/synthetic_v1_seed"))
    parser.add_argument("--records-per-lane", type=int, default=32)
    parser.add_argument("--target-bytes", type=int, default=0)
    parser.add_argument("--target-gb", type=float, default=0.0)
    parser.add_argument("--validation-sample-per-lane", type=int, default=256)
    parser.add_argument("--val-ratio", type=float, default=0.125)
    parser.add_argument("--lanes", nargs="*", default=list(LANES), choices=list(LANES))
    args = parser.parse_args()
    target_bytes = args.target_bytes or int(args.target_gb * 1024 * 1024 * 1024)
    if target_bytes > 0:
        summary = build_synthetic_seed_bundle_to_target(
            output_root=args.output_root,
            target_bytes=target_bytes,
            val_ratio=args.val_ratio,
            lanes=args.lanes,
            validation_sample_per_lane=args.validation_sample_per_lane,
        )
    else:
        summary = build_synthetic_seed_bundle(
            output_root=args.output_root,
            records_per_lane=args.records_per_lane,
            val_ratio=args.val_ratio,
            lanes=args.lanes,
        )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    cli()
