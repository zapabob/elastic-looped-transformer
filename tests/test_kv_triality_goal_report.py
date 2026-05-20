from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_thetom_kv_default_policies_include_turbo8_probe() -> None:
    script = _load_script("thetom_k_protected_kv_sweep")

    assert ("q8_0", "turbo8") in script.DEFAULT_POLICIES
    assert ("bf16", "turbo8") in script.DEFAULT_POLICIES


def test_kv_triality_goal_report_marks_unsupported_turbo8(tmp_path: Path) -> None:
    script = _load_script("render_kv_triality_goal_report")
    kv_dir = tmp_path / "kv"
    kv_dir.mkdir()
    log = kv_dir / "turbo8.log"
    log.write_text('error while handling argument "--cache-type-v": Unsupported cache type: turbo8\n', encoding="utf-8")
    kv_report = kv_dir / "report.json"
    kv_report.write_text(
        json.dumps(
            {
                "raw": [
                    {"policy": "K=q8_0_V=turbo3", "status": "ok", "log": ""},
                    {"policy": "K=q8_0_V=turbo8", "status": "failed", "log": str(log)},
                ],
                "summary": [
                    {
                        "policy": "K=q8_0_V=turbo3",
                        "metric": "gen_tok_s",
                        "n": 1,
                        "mean": 3.0,
                        "sem": 0.0,
                    },
                    {
                        "policy": "K=q8_0_V=turbo3",
                        "metric": "kv_mib",
                        "n": 1,
                        "mean": 2.9,
                        "sem": 0.0,
                    },
                ],
                "pairwise": [
                    {
                        "policy": "K=q8_0_V=turbo3",
                        "metric": "gen_tok_s",
                        "mean_delta": 1.0,
                        "p_value": 0.5,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    audit = tmp_path / "audit"
    audit.mkdir()
    (audit / "triality_so8_rotation_audit_status.json").write_text(
        json.dumps({"status": "pass", "rows": 3, "outliers": 0}),
        encoding="utf-8",
    )
    header = (
        "bits,bit_setting,view,mode,layers,blocks,storage_dtypes,"
        "max_learned_orthogonality_error,mean_learned_orthogonality_error,sd_learned_orthogonality_error,"
        "max_effective_orthogonality_error,mean_effective_orthogonality_error,sd_effective_orthogonality_error,"
        "min_effective_determinant,mean_effective_determinant,max_effective_determinant,"
        "max_effective_determinant_error,mean_effective_determinant_error,sd_effective_determinant_error,status\n"
    )
    body = "".join(
        f"{bits},{bits},vector,fixture,1,1,bfloat16,0.004,0.002,0,0.004,0.002,0,0.996,1.0001,1.004,0.004,0.002,0,pass\n"
        for bits in ("3.0", "4.0", "8.0")
    )
    (audit / "triality_so8_rotation_audit_summary.csv").write_text(header + body, encoding="utf-8")
    metrics = tmp_path / "metrics.jsonl"
    metrics.write_text(
        "\n".join(
            [
                json.dumps({"event": "train_config", "L_max": 3}),
                json.dumps({"event": "train_step", "step": 0, "loss": 1.0, "l_dist": 0.1, "l_entropy": 0.01}),
                json.dumps({"event": "val_probe", "step": 1, "loss": 1.2, "l_dist": 0.2, "l_entropy": 0.02}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    loop_report = tmp_path / "loop.json"
    loop_report.write_text(
        json.dumps(
            {
                "rows": [
                    {"L": 1, "case_id": 0, "correct": 0},
                    {"L": 1, "case_id": 1, "correct": 1},
                    {"L": 2, "case_id": 0, "correct": 1},
                    {"L": 2, "case_id": 1, "correct": 1},
                    {"L": 3, "case_id": 0, "correct": 1},
                    {"L": 3, "case_id": 1, "correct": 1},
                ]
            }
        ),
        encoding="utf-8",
    )

    kv = script.load_kv_report(kv_report)
    triality = script.load_triality_audit(audit)
    ilsd = script.load_ilsd_runs([str(metrics)])
    cv = script.build_loop_aware_cv(loop_report, tmp_path, permutations=100, seed=1)

    turbo8 = next(row for row in kv["rows"] if row["policy"] == "K=q8_0_V=turbo8")
    assert turbo8["ok"] == 0
    assert "Unsupported cache type" in turbo8["unsupported_reason"]
    assert {row["bits"] for row in triality["rows"]} == {3.0, 4.0, 8.0}
    assert ilsd["rows"][0]["max_l_dist"] == 0.2
    assert cv["report"]["n_groups"] == 3
