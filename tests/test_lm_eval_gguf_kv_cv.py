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
