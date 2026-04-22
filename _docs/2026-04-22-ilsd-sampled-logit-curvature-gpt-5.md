# Goal

Add sampled `Δ²-logit` regularization on top of the existing loop-axis entropy stabilizers, while keeping memory bounded on the 1B ELT setup by only reprojecting a small set of uncertain token positions.

# Files Touched

- `C:\Users\downl\Desktop\新しいフォルダー (7)\src\elt_lm\config.py`
- `C:\Users\downl\Desktop\新しいフォルダー (7)\src\elt_lm\ilsd.py`
- `C:\Users\downl\Desktop\新しいフォルダー (7)\src\elt_lm\train.py`
- `C:\Users\downl\Desktop\新しいフォルダー (7)\configs\base_1B.yaml`
- `C:\Users\downl\Desktop\新しいフォルダー (7)\tests\test_ilsd_gradient.py`
- `C:\Users\downl\Desktop\新しいフォルダー (7)\tests\test_smoke_train.py`

# Key Decisions

- Added `logit_curvature_weight` and `logit_curvature_max_positions` to gate a heavier but more direct loop-trajectory penalty.
- Selected sampled positions from uncertainty priority scores derived from teacher entropy and top1-top2 gap.
- Reprojected only the selected token positions from `per_loop_hidden` instead of storing full per-loop logits.
- Kept the sampled `Δ²-logit` term smaller than `Δ²-ent` in the default 1B config.

# Tests

- `uv run --no-sync pyright src/elt_lm/config.py src/elt_lm/ilsd.py src/elt_lm/train.py tests/test_ilsd_gradient.py tests/test_smoke_train.py`
- `uv run --no-sync python -m pytest -q tests/test_ilsd_gradient.py tests/test_smoke_train.py`

# Next Session Notes

- If `L_logit` dominates telemetry early, reduce `logit_curvature_weight` before increasing `max_positions`.
- A further refinement would be sampling with loop argmax-flip information in addition to teacher uncertainty.
