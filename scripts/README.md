# データ準備 → 学習 実行手順 (Windows + H:ドライブ前提)

RTX 3060 12GB / H: 18GB free を想定した縮小パイプライン。全データは `H:\elt_data\` 配下に集約する。

## ディスクレイアウト

```
H:\elt_data\
  raw\                 # HF からの生 JSONL ({"text": ..., "source": ...})
    wiki_ja.jsonl
    wiki_en.jsonl
    metamath.jsonl
    gsm8k.jsonl
    magicoder.jsonl
  bin\                 # Qwen3.5 トークナイザで uint32 パックした学習用
    train.bin
    val.bin
```

## 1. HuggingFace から JA/EN/数学/コード コーパスを H: に収集

```bash
# デフォルト: 合計 ~5.8GB raw JSONL まで (wiki_ja/wiki_en 各 2.5GB, metamath 0.5GB, ...)
uv run python scripts/download_hf_corpus.py --out H:/elt_data/raw

# スモーク (全て 10%): ~600MB
uv run python scripts/download_hf_corpus.py --out H:/elt_data/raw --scale 0.1

# 特定ソースだけ
uv run python scripts/download_hf_corpus.py --out H:/elt_data/raw --only wiki_ja metamath
```

収集対象:

| name | dataset | 用途 | デフォルト上限 |
|---|---|---|---|
| wiki_ja | `range3/wikipedia-ja-20230101` | 日本語一般知識 | 2.5GB |
| wiki_en | `wikimedia/wikipedia 20231101.en` | 英語一般知識 | 2.5GB |
| metamath | `meta-math/MetaMathQA` | 数学 CoT | 0.5GB |
| gsm8k | `openai/gsm8k main` | 算数 CoT | (全量) |
| magicoder | `ise-uiuc/Magicoder-OSS-Instruct-75K` | コード指示 | 0.3GB |

HF へのログインが必要なデータセットがあれば `huggingface-cli login` を先に。

## 2. ローカル資産 + HF DL 済みを統合してトークナイズ

マニフェスト `scripts/corpus_manifest.yaml` は既にローカル aegis・wikipedia/*.txt・`H:\from_D\dataset\final\*.jsonl` と HF 収集分を統合する形で書かれている。

```bash
uv run python scripts/build_train_bin.py \
  --tokenizer H:/Qwen3.5-9B-official-hf \
  --out-dir   H:/elt_data/bin \
  --config    scripts/corpus_manifest.yaml
```

生成物: `H:\elt_data\bin\train.bin` (98%) + `val.bin` (2%)。
これは `configs/tiny_10M.yaml` / `configs/base_100M.yaml` 内の `data.train_bin / val_bin` と一致する。

### スキーマ自動判定

| `type` | 期待形式 |
|---|---|
| `flat` | JSONL 各行 `{"text": "..."}` |
| `aegis_reasoning` | aegis CoT 形式 (`{"id","text","category",...}`) — 単純に `text` を取り出す |
| `aegis_sft` | aegis SFT 形式 (`{"instruction": "<str(dict)>"}`) — 内部の `messages` を `<\|role\|>\ncontent` で結合して復元 |
| `txt` | utf-8 プレーンテキスト1ファイル / ディレクトリ再帰 |

path がディレクトリならその配下の `*.jsonl` (または `*.txt`) を再帰 glob。

## 3. 動作検証 (tiny 10M)

```bash
uv run elt-train --config configs/tiny_10M.yaml
```

数十分～数時間。損失が単調減少し ILSD の 3 項 (`L_GT_t, L_GT_s, L_dist`) すべて finite であることを確認。

## 4. 本命 (base 100M, RTX 3060)

```bash
uv run elt-train --config configs/base_100M.yaml
```

bf16 + grad-ckpt + micro-bs 2 × accum 16 = effective 32 で ~2-5 日。

## 5. GRPO 後学習 (GSM8K 例)

事前学習 (+ SFT) 済みの checkpoint に GRPO を当てる流れ:

```bash
# 5.1 GSM8K プロンプトを生成 (<think>/<answer> 指示つき + 参照は `#### N`)
uv run python scripts/build_grpo_prompts.py \
  --in  H:/elt_data/raw/gsm8k.jsonl \
  --out H:/elt_data/grpo/gsm8k_prompts.jsonl

# 5.2 GRPO 訓練 (1 プロンプトあたり G=8 ロールアウト, KL β=0.05, clip ε=0.2)
uv run elt-train-grpo --config configs/grpo_gsm8k.yaml
```

`configs/grpo_gsm8k.yaml` の `grpo.init_ckpt` に SFT 済み `.pt` を指定する。
`verifiers.py` の `correct × format` 乗算ゲートと `length/repeat` 負値加算で
報酬ハックを遮断し、`grpo.py` の DeepSeek 非バイアス KL (`exp(d) - d - 1`) で
参照方策からの逸脱にコストを課す。

## 6. Any-Time 評価

```bash
uv run elt-anytime --ckpt runs/base_100M/last.pt --val-bin H:/elt_data/bin/val.bin --L-list 1,2,3,4
```

L ∈ [L_min, L_max] それぞれの NLL / PPL / 相対 FLOPs を CSV 出力。論文 Fig 相当の Any-Time 曲線。
