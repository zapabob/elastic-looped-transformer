---
date: 2026-04-21
slug: hot-path-integration
ai: opus-4-7
---

# Phase-C 追補: 学習ループ統合 + 1B/300M fit 実測

前回コミット (`nvme-offload`) で TieredParameterStore / NvmeAdamW / prefetcher / placement
の部品は揃ったが、`train.train()` から実際に NvmeAdamW を構築する経路と、その配線を
1 呼び出しで済ませるヘルパが未実装だった。本セッションで統合 + 実測まで実施。

## 目的

- `configs/*.yaml` の `optim.kind: nvme_adamw` を学習ループから呼び出せるようにする
- 1B モデルが RTX 3060 12 GB VRAM に収まることを PagedAdamW8bit / NvmeAdamW の両方で
  実際に走らせて測る
- 300M smoke を H: ドライブ (20 GB 空き) で回して NvmeAdamW のデータパス全体を
  (allocate → read → CPU step → write) 通して壊れないことを示す

## 実装

### `src/elt_lm/offload/hooks.py` (新規)

- `LayerTimingInstrumentor`: CompositeBlock の各層に forward pre/post hook を張り、
  `layer_computed` テレメトリを発行 (`layer_idx`, `tier`, `duration_us`)。ダッシュボード
  Storage-tiers パネルが消費する。
- `install_offload_into_training(model, cfg, run_dir)`: `probe_hardware` → `plan_placement`
  → `TieredParameterStore` → `NvmeAdamW` までを一括で行い `(opt, store)` を返す
  1-liner。weight_decay の 2D-only 慣行は `train.configure_optimizer` と揃えた。

### 設計上の判断: CompositeBlock.forward に promote/demote は入れない

当初プラン (§2.2) では layer weight の bf16 コピーを RAM に置いて、forward 毎に
RAM → pinned → GPU と昇格させる設計だった。実際に `NvmeAdamW` を書いたところ:

- 学習中の params は GPU 常駐。NVMe に置くのは `m`, `v`, `master_fp32` の **optimizer state** のみ
- `NvmeAdamW.step()` は CPU 上で `bf16 grad → fp32 master update → bf16 writeback` を
  完結し、forward/backward 側は通常の torch モデルとして動く
- 「forward 毎の promote/demote」を入れても optimizer state が NVMe にある限り
  step レイテンシは変わらず、forward コストが純増する

したがって hooks.py は **観測のみ** の目的に絞り、weight の物理移動は挿入しない。
`.md` のプランと実装の差異をここに明記 (`offload/hooks.py` 冒頭 docstring にも)。

### `src/elt_lm/train.py` 配線

`train()` 内で `cfg.optim.kind == "nvme_adamw"` のときだけ
`install_offload_into_training(model, cfg=cfg, run_dir=run_dir)` を呼び、返却された
`offload_store` を `finally` 節で `flush()` する。他の kind は従来どおり
`configure_optimizer(model, cfg)`。

### `configs/smoke_300M.yaml` (新規)

H: 20 GB 空きに収まる 300M-class config。`d_model=1024, N=16, d_ff=2816`,
non-emb 205 M → fp32 state ≈ 2.5 GB。1B まるごとだと NvmeAdamW state が 13 GB で
H: がギリギリ埋まるため、end-to-end 挙動検証はこの config で行う。

### `scripts/smoke_1b_vram.py`

`--config` 追加で任意 yaml を指定可能に。`--optim {adamw,paged_adamw_8bit,nvme_adamw}`
を選んで 1 step 回し peak VRAM / elapsed を出力。

### `tests/test_offload_hooks.py` (4 テスト)

- `LayerTimingInstrumentor` が unique_layers × L 個のイベントを発行
- コンテキスト脱出で hook が外れる (`_forward_hooks`, `_forward_pre_hooks` 空)
- `install_offload_into_training` が `NvmeAdamW` + `TieredParameterStore` を返す
- fwd+bwd+NvmeAdamW.step() 通しで NaN なし

## 実測

### PagedAdamW8bit × `configs/base_1B.yaml`

```
uv run python scripts/smoke_1b_vram.py --optim paged_adamw_8bit
```

- params total 1.537 B / non-emb 1.092 B
- **peak VRAM 7.88 GB** (fits 12 GB ✓)
- step time ~5.0 s (初回 cuDNN autotune 含む)

→ 1B 事前学習の本命経路として確定。

### NvmeAdamW × `configs/smoke_300M.yaml`

```
uv run python -u scripts/smoke_1b_vram.py --optim nvme_adamw \
  --config configs/smoke_300M.yaml \
  --run-dir H:/elt_data/runs/smoke_300M_nvme
```

- params total 0.460 B / non-emb 0.206 B
- build: 5.5 s / optimizer alloc: 3.2 s (NVMe shard 432 本 = 16 層 × 9 param × 3 shard)
- post-build VRAM: 0.87 GB
- **fwd+bwd+step 128.7 s** (step 内で全 non-emb param を CPU → NVMe 往復するコスト)
- **peak VRAM 4.38 GB** (fits 12 GB ✓)
- loss 12.60 (未学習 300M + ランダムトークンとして妥当)

→ データパスは通っている。grad_accum を効かせれば step 頻度は下げられるが、
NvmeAdamW は 1B 本番学習用というより「VRAM 優先、時間度外視」の予備経路という位置付け。
本命は PagedAdamW8bit。

### 1B × NvmeAdamW は未実施

fp32 state ~13 GB (m+v+master の 4 bytes × 1.09 B × 3)。H: の空きが 20 GB のため
本番 runs や他データと共存するには窮屈。1B NvmeAdamW smoke は H: 整理後の宿題とする。

## 触ったファイル

- 新規
  - `src/elt_lm/offload/hooks.py`
  - `configs/smoke_300M.yaml`
  - `scripts/smoke_1b_vram.py`
  - `tests/test_offload_hooks.py`
  - `_docs/2026-04-21-hot-path-integration-opus-4-7.md`
- 編集
  - `src/elt_lm/train.py` (nvme_adamw 分岐 + store.flush)

## 次セッションへの引き継ぎ

- H: を整理して 1B NvmeAdamW 実測 (peak VRAM / step time)
- 本番事前学習は `configs/base_1B.yaml` + `optim.kind: paged_adamw_8bit` で起動
- 1B ランは pipeline.py からでなく個別に `uv run elt-train --config configs/base_1B.yaml`
  で走らせ、ダッシュボードで telemetry を見ながら数時間〜数日観察
