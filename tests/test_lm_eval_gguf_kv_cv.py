from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load_script():
    spec = importlib.util.spec_from_file_location(
        "run_lm_eval_gguf_kv_cv",
        ROOT / "scripts" / "run_lm_eval_gguf_kv_cv.py",
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_build_lm_eval_rows_creates_letter_choice_cv_rows(tmp_path: Path) -> None:
    script = _load_script()
    source = tmp_path / "heldout.jsonl"
    source.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "prompt": "Question 1\nChoices:\nA. a\nB. b\nC. c\nD. d",
                        "reference": "B",
                        "source": "hf:test",
                        "metadata": {"task_name": "toy_stem"},
                    }
                ),
                json.dumps(
                    {
                        "prompt": "Question 2\nChoices:\nA. a\nB. b\nC. c\nD. d",
                        "reference": "D",
                        "source": "hf:test",
                        "metadata": {"task_name": "toy_stem"},
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    rows = script.build_lm_eval_rows(source, folds=2)

    assert [row["fold"] for row in rows] == [0, 1]
    assert rows[0]["choices"] == [" A", " B", " C", " D"]
    assert rows[0]["target"] == 1
    assert rows[1]["target"] == 3
    assert rows[0]["prompt"].endswith("\n\nAnswer:")


def test_build_lm_eval_rows_converts_gsm8k_to_numeric_mcq(tmp_path: Path) -> None:
    script = _load_script()
    source = tmp_path / "gsm8k.jsonl"
    source.write_text(
        json.dumps(
            {
                "prompt": "Solve it.\n\nQuestion:\nA pack has 40 cards. Kim gives away 18. How many remain?",
                "reference": "Kim has 40 - 18 = 22 cards.\n#### 22",
                "task": "gsm8k",
                "source": "hf:gsm8k/main/test",
                "metadata": {"task_name": "gsm8k", "benchmark": "gsm8k"},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    rows = script.build_lm_eval_rows(source, folds=4)

    assert len(rows) == 1
    assert rows[0]["benchmark"] == "gsm8k_numeric_mcq"
    assert rows[0]["choices"] == [" A", " B", " C", " D"]
    assert rows[0]["choice_values"][rows[0]["target"]] == "22"
    assert "Choices:" in rows[0]["prompt"]
    assert rows[0]["prompt"].endswith("\n\nAnswer:")


def test_find_label_logprob_accepts_spaced_and_unspaced_tokens() -> None:
    script = _load_script()
    top = [
        {"token": "A", "logprob": -3.0},
        {"token": " B", "logprob": -0.5},
    ]

    assert script.find_label_logprob(top, " B") == -0.5
    assert script.find_label_logprob(top, "B") == -0.5
    assert script.find_label_logprob(top, " C") is None


def test_public_command_redacts_local_runtime_paths() -> None:
    script = _load_script()
    command = [
        "C:/Users/example/AppData/Local/Programs/llama-turboquant/bin/llama-server.exe",
        "-m",
        "H:/elt_data/releases/model.gguf",
        "--cache-type-v",
        "turbo3",
    ]

    redacted = script.public_command(
        command,
        llama_server=Path(command[0]),
        model=Path(command[2]),
    )

    assert redacted[0] == "llama-server.exe"
    assert redacted[2] == "model.gguf"
    assert "Users" not in " ".join(redacted)


def test_compare_policies_keeps_fold_pairing_for_p_values() -> None:
    script = _load_script()
    rows = [
        {"policy": "K=q8_0_V=turbo3", "fold": 0, "correct": 1},
        {"policy": "K=q8_0_V=turbo3", "fold": 0, "correct": 1},
        {"policy": "K=q8_0_V=turbo3", "fold": 1, "correct": 0},
        {"policy": "K=q8_0_V=turbo3", "fold": 1, "correct": 1},
        {"policy": "K=bf16_V=turbo3", "fold": 0, "correct": 1},
        {"policy": "K=bf16_V=turbo3", "fold": 0, "correct": 0},
        {"policy": "K=bf16_V=turbo3", "fold": 1, "correct": 0},
        {"policy": "K=bf16_V=turbo3", "fold": 1, "correct": 0},
    ]

    comparison = script.compare_policies(rows, folds=2)

    assert comparison["fold_scores"]["K=q8_0_V=turbo3"] == [1.0, 0.5]
    assert comparison["fold_scores"]["K=bf16_V=turbo3"] == [0.5, 0.0]
    assert comparison["summaries"][0]["mean"] == 0.75
    assert comparison["pairwise"][0]["left"] == "K=q8_0_V=turbo3"
    assert comparison["pairwise"][0]["right"] == "K=bf16_V=turbo3"
    assert 0.0 <= comparison["pairwise"][0]["p_value"] <= 1.0


def test_compare_policies_by_benchmark_slices_rows() -> None:
    script = _load_script()
    rows = [
        {"policy": "K=q8_0_V=turbo3", "fold": 0, "correct": 1, "benchmark": "mmlu_stem"},
        {"policy": "K=q8_0_V=turbo3", "fold": 1, "correct": 0, "benchmark": "mmlu_stem"},
        {"policy": "K=bf16_V=turbo3", "fold": 0, "correct": 1, "benchmark": "mmlu_stem"},
        {"policy": "K=bf16_V=turbo3", "fold": 1, "correct": 1, "benchmark": "mmlu_stem"},
        {"policy": "K=q8_0_V=turbo3", "fold": 0, "correct": 0, "benchmark": "gsm8k_numeric_mcq"},
        {"policy": "K=q8_0_V=turbo3", "fold": 1, "correct": 0, "benchmark": "gsm8k_numeric_mcq"},
        {"policy": "K=bf16_V=turbo3", "fold": 0, "correct": 0, "benchmark": "gsm8k_numeric_mcq"},
        {"policy": "K=bf16_V=turbo3", "fold": 1, "correct": 1, "benchmark": "gsm8k_numeric_mcq"},
    ]

    by_benchmark = script.compare_policies_by_group(rows, folds=2, group_key="benchmark")

    assert sorted(by_benchmark) == ["gsm8k_numeric_mcq", "mmlu_stem"]
    assert by_benchmark["mmlu_stem"]["summaries"][0]["mean"] == 0.5
    assert by_benchmark["gsm8k_numeric_mcq"]["summaries"][0]["mean"] == 0.0
