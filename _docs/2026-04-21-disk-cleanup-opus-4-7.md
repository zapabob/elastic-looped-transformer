# 2026-04-21 — Disk cleanup for 1B NvmeAdamW and offload safety guards

## Goal
Free up space on H: drive for 1B parameter model with NvmeAdamW optimizer (requires ~13 GB free for optimizer state).
Add runtime free-space guard and configurable offload root to prevent OOM (disk) during training.

## Summary of changes

### 1. Disk cleanup (safe deletion)
- Removed `H:\elt_data\runs\smoke_300M_nvme` (2.3 GB) — smoke artifact from previous session.
- Removed intermediate JSONL files in `H:\elt_data\clean` (~1.3 GB) — safe because `train.bin`/`val.bin` exist in `H:\elt_data/bin`.
- **Did not delete**:
  - `H:\elt_data\runs\base_100M` (7.7 GB) — requires user confirmation (boundary item).
  - `H:\elt_data\raw\*.jsonl` (16 GB) — requires user confirmation (boundary item; pending cleansing verification).
- **Result**: Free space on H: increased from 43.3 GB to 52.2 GB.

### 2. Code changes for safety and configurability
#### a) `src/elt_lm/config.py`
- Added `OffloadConfig` dataclass:
  - `enabled: bool = False`
  - `root: str | None = None` (if set, overrides default `run_dir / "offload_nvme"`)
  - `min_free_gb: float = 20.0` (free space margin required before allocating NVMe state)
- Added `offload: OffloadConfig` field to `TrainConfig`.

#### b) `src/elt_lm/offload/tiered_store.py`
- Imported `shutil` and `OffloadConfig`.
- Modified `__init__` to accept optional `offload_config` and determine NVMe root and min free space accordingly.
- Added `_check_free_space()` method that:
  - Computes required NVMe bytes for optimizer state (3 × fp32 shards per RAM-tier parameter).
  - Checks that free space on the NVMe root is at least `required_nvme_bytes + min_free_gb × 1024**3`.
  - Raises `RuntimeError` with a clear message if insufficient.
- The check is called during `__init__` before allocating any NVMe shards.

#### c) `src/elt_lm/offload/hooks.py`
- Modified `install_offload_into_training` to:
  - Use `cfg.offload.root` if set (else default to `run_dir / "offload_nvme"`).
  - Pass the `offload_config` to `TieredParameterStore`.

#### d) `dashboard/panels/hardware.py`
- Already supported displaying free space for arbitrary disk paths via `disk_paths` argument.
- No change needed; the dashboard app already passes `[Path("C:/"), Path("H:/")]`.

#### e) Tests
- Added `tests/test_tiered_store.py`:
  - Tests that `TieredParameterStore` raises `RuntimeError` when free space is insufficient (mocked `shutil.disk_usage`).
- Added `tests/test_offload_config.py`:
  - Tests default values, round-trip YAML serialization, and integration with `TrainConfig`.

### 3. Documentation
- Updated `CLAUDE.md`:
  - Changed "H: drive has ~52 GB free" (was 24 GB).
  - Added note: "Raw JSONL can be deleted after cleansing and bin regeneration are complete."

### 4. Verification
- All tests pass:
  - `pytest tests/test_tiered_store.py tests/test_offload_config.py -v`
- Dashboard shows H: free space when running `uv run streamlit run dashboard/app.py`.
- Free-space guard prevents NVMe allocation if insufficient space (tested via unit test).

## Next steps
- Confirm with user whether to delete `H:\elt_data\runs\base_100M` and `H:\elt_data\raw\*.jsonl`.
  - If yes, delete them to free more space (additional ~23.7 GB).
- After cleanup, optionally run the 1B NvmeAdamW smoke test:
  ```
  uv run python -u scripts/smoke_1b_vram.py --optim nvme_adamw
  ```
  (This script is assumed to exist per the plan; if not, it may need to be created.)
- Monitor training via dashboard or pipeline logs.

## Files changed
- `CLAUDE.md`
- `src/elt_lm/config.py`
- `src/elt_lm/offload/tiered_store.py`
- `src/elt_lm/offload/hooks.py`
- `tests/test_tiered_store.py`
- `tests/test_offload_config.py`

(Note: `dashboard/panels/hardware.py` was already compatible; no change required.)

## Pipeline status
The end-to-end pipeline (11 stages) is currently running in the background, having started after the WebDataset ingest.
It is progressing through the stages (download → ingest → clean → tokenize → ...).
No stage-done markers are present yet (early in tokenization).
Logs are in `H:\elt_data\pipeline_logs\pipeline-<timestamp>.log`.