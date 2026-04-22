# Goal

Promote loop-axis entropy stability to the primary ELT auxiliary regularizer by adding entropy second-difference (`Δ²-ent`) and uncertainty-weighted loop entropy floor, while keeping hidden-state local consistency as a lighter supplement.

# Files Touched

- `C:\Users\downl\Desktop\新しいフォルダー (7)\src\elt_lm\config.py`
- `C:\Users\downl\Desktop\新しいフォルダー (7)\src\elt_lm\ilsd.py`
- `C:\Users\downl\Desktop\新しいフォルダー (7)\src\elt_lm\train.py`
- `C:\Users\downl\Desktop\新しいフォルダー (7)\configs\base_1B.yaml`
- `C:\Users\downl\Desktop\新しいフォルダー (7)\tests\test_ilsd_gradient.py`
- `C:\Users\downl\Desktop\新しいフォルダー (7)\tests\test_smoke_train.py`

# Key Decisions

- Reused `per_loop_hidden` and reprojected inside `ilsd.py` to avoid storing full per-loop logits.
- Added `entropy_curvature_weight` as the main loop-trajectory smoothness term.
- Switched entropy floor from student-only to loop-aware when per-loop hidden states are available.
- Added uncertainty-based token weighting from teacher entropy plus top1-top2 probability gap.
- Kept hidden-state local consistency, but reduced its recommended default weight in `base_1B`.

# Tests

- `uv run --no-sync pyright src/elt_lm/config.py src/elt_lm/ilsd.py src/elt_lm/train.py tests/test_ilsd_gradient.py tests/test_smoke_train.py`
- `uv run --no-sync python -m pytest -q tests/test_ilsd_gradient.py tests/test_smoke_train.py`

# Next Session Notes

- If training telemetry shows the curvature term dominating too early, lower `entropy_curvature_weight` before changing the floor schedule.
- The next logical extension is sampled `Δ²-logit` for only uncertain token positions, not dense full-vocab loop logits.
