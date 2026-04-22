from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml

from elt_lm.config import DataConfig, ModelConfig, TrainConfig
from elt_lm.eval.anytime_sweep import run
from elt_lm.eval.benchmarks import (
    BenchmarkSpec,
    evaluate_benchmark,
    load_benchmark_cases,
    load_benchmark_manifest,
    multiple_choice_correctness,
    score_response,
)
from elt_lm.model import ELTLanguageModel


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def test_load_benchmark_manifest_and_cases_jsonl(tmp_path: Path) -> None:
    cases_path = tmp_path / "cases.jsonl"
    _write_jsonl(cases_path, [
        {"prompt": "Q1", "reference": "A1"},
        {"prompt": "Q2", "reference": "A2"},
    ])
    manifest_path = tmp_path / "benchmarks.yaml"
    manifest_path.write_text(yaml.safe_dump({
        "benchmarks": [{
            "name": "smoke_exact",
            "kind": "jsonl",
            "task": "exact_match",
            "path": str(cases_path),
            "prompt_field": "prompt",
            "reference_field": "reference",
        }],
    }), encoding="utf-8")

    specs = load_benchmark_manifest(manifest_path)
    assert len(specs) == 1
    loaded = load_benchmark_cases(specs[0])
    assert [c.prompt for c in loaded] == ["Q1", "Q2"]
    assert [c.reference for c in loaded] == ["A1", "A2"]


def test_score_response_supports_multiple_choice_and_json() -> None:
    assert multiple_choice_correctness("I pick B", "B") == 1.0
    assert score_response("json_match", '{"a": 1}', '{"a": 1}') == 1.0
    assert score_response("gsm8k", "The answer is 42", "#### 42") == 1.0


class _FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 99
    eos_token = "<eos>"
    pad_token = "<pad>"

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return [len(text) % 11 + 1]

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        del skip_special_tokens
        if ids == [7]:
            return "A"
        return " ".join(str(i) for i in ids)


def test_anytime_sweep_runs_benchmark_manifest(tmp_path: Path, monkeypatch) -> None:
    ckpt = tmp_path / "tiny.pt"
    cfg = TrainConfig(
        model=ModelConfig(
            vocab_size=128,
            d_model=32,
            n_unique_layers=2,
            n_heads=2,
            head_dim=16,
            d_ff=64,
            max_seq_len=32,
            tie_word_embeddings=True,
            grad_checkpoint=False,
            L_min=1,
            L_max=2,
        ),
        data=DataConfig(tokenizer_path="unused"),
    )
    model = ELTLanguageModel(cfg.model)
    torch.save({"cfg": cfg, "model": model.state_dict()}, ckpt)

    bench_cases = tmp_path / "bench.jsonl"
    _write_jsonl(bench_cases, [{"prompt": "Choose one", "reference": "A"}])
    manifest = tmp_path / "bench.yaml"
    manifest.write_text(yaml.safe_dump({
        "benchmarks": [{
            "name": "choice_smoke",
            "kind": "jsonl",
            "task": "exact_match",
            "path": str(bench_cases),
            "prompt_field": "prompt",
            "reference_field": "reference",
        }],
    }), encoding="utf-8")

    def _fake_from_pretrained(*_args, **_kwargs):
        return _FakeTokenizer()

    def _fake_generate(self, input_ids, **_kwargs):
        resp = torch.tensor([[7]], dtype=input_ids.dtype, device=input_ids.device)
        return torch.cat([input_ids, resp], dim=-1)

    monkeypatch.setattr("transformers.AutoTokenizer.from_pretrained", _fake_from_pretrained)
    monkeypatch.setattr(ELTLanguageModel, "generate", _fake_generate)

    run_dir = tmp_path / "run"
    args = argparse.Namespace(
        ckpt=str(ckpt),
        val_bin="",
        benchmark_manifest=str(manifest),
        seq_len=0,
        batch_size=1,
        max_batches=1,
        bench_max_new_tokens=8,
        bench_temperature=0.0,
        bench_top_k=1,
        bench_num_samples=1,
        bench_verifier_retries=0,
        out_csv=str(tmp_path / "bench.csv"),
        run_dir=str(run_dir),
    )
    run(args)

    events = [
        json.loads(line)
        for line in (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    bench_events = [e for e in events if e["event"] == "benchmark_eval"]
    assert len(bench_events) == 2
    assert {int(e["L"]) for e in bench_events} == {1, 2}
    assert all(abs(float(e["score"]) - 1.0) < 1e-6 for e in bench_events)
    assert all(abs(float(e["attempts_per_case"]) - 1.0) < 1e-6 for e in bench_events)


def test_evaluate_benchmark_retries_until_verifier_passes(monkeypatch) -> None:
    cfg = TrainConfig(
        model=ModelConfig(
            vocab_size=128,
            d_model=32,
            n_unique_layers=2,
            n_heads=2,
            head_dim=16,
            d_ff=64,
            max_seq_len=32,
            tie_word_embeddings=True,
            grad_checkpoint=False,
            L_min=1,
            L_max=2,
        ),
        data=DataConfig(tokenizer_path="unused"),
    )
    model = ELTLanguageModel(cfg.model)
    tok = _FakeTokenizer()
    spec = BenchmarkSpec(
        name="retry_exact",
        kind="jsonl",
        task="exact_match",
        path=None,
        prompt_field="prompt",
        reference_field="reference",
    )

    responses = iter([torch.tensor([[5]]), torch.tensor([[7]])])

    def _fake_generate(self, input_ids, **_kwargs):
        resp = next(responses).to(device=input_ids.device, dtype=input_ids.dtype)
        return torch.cat([input_ids, resp], dim=-1)

    def _fake_cases(_spec):
        return [type("Case", (), {
            "prompt": "Choose one",
            "reference": "A",
            "task": "exact_match",
            "benchmark": "retry_exact",
        })()]

    monkeypatch.setattr(ELTLanguageModel, "generate", _fake_generate)
    monkeypatch.setattr("elt_lm.eval.benchmarks.load_benchmark_cases", _fake_cases)

    result = evaluate_benchmark(
        model=model,
        tokenizer=tok,
        spec=spec,
        L=1,
        device=torch.device("cpu"),
        max_new_tokens=8,
        temperature=0.8,
        top_k=10,
        num_samples=1,
        verifier_retries=1,
    )

    assert result.correct == 1
    assert result.total == 1
    assert abs(result.attempts_per_case - 2.0) < 1e-6
