---
date: 2026-04-21
slug: disk-guard-readme
ai: opus-4-7
---

# H: 空き確保 + OffloadConfig YAML + README 全面リライト

## 目的

1B NvmeAdamW の本番起動に向け、H: の空き確保 / config 経由の offload_root 切替 /
README を 1B + offload + dashboard 込みに刷新する。

## ディスク整理 (H:)

- `H:\elt_data\runs\smoke_300M_nvme` (2.3 GB) は前回セッションの smoke 成果物。削除済み。
- `H:\elt_data\runs\base_100M` は既に存在せず (事前削除済みだった)。
- `H:\elt_data\clean` は空。`bin/` には `train.bin (1.3 GB)` と `val.bin (27 MB)` が健在。
- **H: 空き: 43.3 GB → 62.3 GB** (fsutil 実測、Get-PSDrive は別値を返すので fsutil を信頼)。
- 1B NvmeAdamW fp32 state (13 GB) + rolling ckpt + accum 余裕で収まる。

## コード / 設定変更

### `src/elt_lm/config.py`

`load_train_config()` が YAML の `offload:` ブロックをパースしていなかったので
`OffloadConfig(**offload_raw)` を追加。これがないと `cfg.offload.root` が常に
None になっていた (回帰)。

### `src/elt_lm/offload/tiered_store.py`

`offload_config is None` のときの `min_free_gb` デフォルトを **20.0 → 0.0** に変更。
理由: テスト・ad-hoc スクリプトは OffloadConfig を渡さず `nvme_root` だけを使うので、
20 GB マージンが C: の tmp_path に刺さって全滅する。**production 経路 (install_offload)
は OffloadConfig を必ず渡す**ので実害なし。guard の検出範囲は OffloadConfig の
min_free_gb に集中させる。

### `.gitignore`

`*.f32`, `offload_nvme/`, `logs/` を追加。NvmeAdamW の memmap シャードと smoke ログが
うっかり commit されないように。

### README.md (全面リライト)

旧 README は 85M スケール限定記述だったので、現状の

- 4 スケール (10M / 100M / 300M / 1B) + 実測数値
- PagedAdamW8bit で 1B が 12 GB に乗る (**peak 7.88 GB**)
- NvmeAdamW の配線と使い方 (`offload.root` / `min_free_gb`)
- dashboard 6 パネル + テレメトリ JSONL
- Install 群に `--extra offload_8bit / dashboard / dev`

badge の tests を `105 → 110/110`、scales badge を `10M | 100M | 300M | 1B` に更新。

### CLAUDE.md

空き注記を 2026-04-21 の実測に合わせて更新 (~62 GB 空き、
NvmeAdamW state は `cfg.offload.root` 配下、1B で 13 GB バジェット)。

## テスト

- `tests/test_offload_config.py`:
  - `test_load_train_config_parses_offload_section` — YAML `offload:` が反映される
  - `test_load_train_config_offload_missing_block` — ブロック無しでもデフォルトが通る
  - `test_tiered_store_raises_when_disk_full` — `shutil.disk_usage` をモックして
    RuntimeError を確認
- `tests/test_offload_hooks.py`:
  - 既存の 2 件で `OffloadConfig(min_free_gb=0.0)` を明示 (tmp_path が C: にあるため)
- `tests/test_optim_config.py`:
  - `nvme_adamw` の期待例外を `NotImplementedError` → `RuntimeError("install_offload_into_training")`
    に更新 (commit fbb6283 のコード変更に合わせた回帰修正)

`pytest -q` = **110 passed**。

## 触ったファイル

- 編集: `.gitignore`, `CLAUDE.md`, `README.md`, `src/elt_lm/config.py`,
  `src/elt_lm/offload/tiered_store.py`, `tests/test_offload_hooks.py`,
  `tests/test_optim_config.py`
- 新規: `tests/test_offload_config.py`, `_docs/2026-04-21-disk-cleanup-opus-4-7.md`
  (先行セッションで作成済み), `_docs/2026-04-21-disk-guard-readme-opus-4-7.md` (本ログ)

## 次セッションへの引き継ぎ

- `cfg.offload.enabled` はまだ意味を持たない (train ループは `cfg.optim.kind` で分岐)。
  `enabled` と `kind=nvme_adamw` の整合はドキュメントでは「両方立てる」ことを推奨。
  将来的に `enabled` を真のゲート化するか、`kind` にまとめるか検討。
- 1B × NvmeAdamW smoke は H: 62 GB で回せる。`uv run python -u scripts/smoke_1b_vram.py
  --optim nvme_adamw` で実測。
- `H:\elt_data\raw` (16 GB) は task 21 (cleansing→bin) が完了しているか確認の上、
  削除判断をユーザーに委ねる。
