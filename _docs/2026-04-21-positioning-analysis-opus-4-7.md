---
date: 2026-04-21
slug: positioning-analysis
ai: opus-4-7
---

# 既存ローカル LLM との性能比較・elt-lm のポジショニング分析

## 目的

1B Phase-1 pretrain の目処が立った段階で、「結局どのくらいのモデルになりうるか」を
同規模 (1B–2B) の公開ベンチ数値と突き合わせて**正直に**言語化しておく。
README の Roadmap / 論文の contribution 節 / 外向き広報 (X / LinkedIn)
すべての共通基盤になる判断材料として残す。

- 対象: elt-lm-base-1.5B (1.09B non-emb / 1.54B total / N=28 / L=4 → 実効 112 層)
- 学習予算: 現 DL-2 (task 32 継続中) で ~5–6B tokens、全完了で ~10B tokens 見込み
- 比較軸: MMLU (総合/STEM) / GSM8K / HumanEval / BFCL / LiveCodeBench

## 1. 汎用ベンチ (MMLU / GSM8K / HumanEval)

### 同規模 OSS のリファレンス

| モデル | params | 学習 tokens | MMLU | GSM8K | HumanEval |
|---|---|---|---|---|---|
| TinyLlama-1.1B | 1.1B | 3T | ~26 | ~3 | ~10 |
| Llama-3.2-1B | 1.24B | 9T | ~49 | ~44 (IT) | — |
| Qwen2.5-1.5B | 1.5B | 18T | ~60 | ~68 | ~37 |
| SmolLM2-1.7B | 1.7B | **11T** | ~50 | ~31 | — |
| Phi-3.5-mini | 3.8B | 3.4T | ~69 | **~86** | — |

### elt-lm-base-1.5B の予測 (段階別)

| 段階 | MMLU 総合 | MMLU-STEM | GSM8K | HumanEval |
|---|---|---|---|---|
| Phase-1 base のみ | 30–40 | 40–50 | 5–15 | 10–20 |
| + Phase-2 SFT (CoT) | 40–50 | 50–60 | 20–35 | 20–30 |
| + Phase-3 GRPO + L=4 推論 | 40–50 | **55–65** | **35–55** | **25–40** |

### 所感

- **MMLU-STEM は Qwen2.5-1.5B (60) 近辺まで行ける可能性**: ArXiv + STEM 重点コーパス
  (task 24) + Phi-3.5 系の「質は量を部分的に代替する」存在証明より。
- **MMLU 総合は 40–50 台**: humanities / law / social science が相対的に薄くなる想定。
  コーパスバイアスで「STEM 強・人文弱」な形になる。
- **GSM8K は GRPO の `correct × format` verifier が直接効く**。L=4 の追加計算と
  組み合わせて 35–55%。Llama-3.2-1B-Instruct (44) を超える余地は十分ある。
  Qwen2.5-1.5B (68) には届かないが、同一土俵で張り合える帯には乗る。
- **Phase-1 base の GSM8K が 5–15% と低いのは、CoT 発話能力の未獲得が主因**。
  pretrain のみで 44% に届く base モデルは存在しない (Llama-3.2-1B の 44 は Instruct 版)。

## 2. L-iteration が効く場面・効かない場面

Universal Transformer 系 (Dehghani+18, Csordás+21) の実証より:

| 伸びる | 伸びない |
|---|---|
| 算術 (加減乗除、桁上げ) | MMLU の知識問題 (weight-bound) |
| 多段論理 (entailment chain) | 固有名詞・年号の recall |
| アルゴリズム系 (sort, parse, tree traverse) | 文化・常識知識 |
| CoT 形式の推論 | 翻訳品質 |

**原則**: L-iteration は「per-token の計算量を増やす」。知識密度は重み容量律速なので
L=4 でも伸びない。GSM8K / HumanEval / 多段 ReAct が伸びる理由はこれ。

## 3. ツールコール・汎用エージェント

### 同規模ベースライン

| タスク層 | 代表ベンチ | 1B 級の到達水準 |
|---|---|---|
| 単発 function call (1 ツール) | BFCL Simple / Nexus | **xLAM-1B: 40–65% 実証済み** |
| 多ツール選択・parallel 呼出 | BFCL Multiple/Parallel | Llama-3.2-1B: 26 / 3B: 67 (**段差大**) |
| ReAct ループ (2–5 step) | ToolBench | 15–30% |
| 汎用エージェント | GAIA / WebArena / SWE-bench | ~0% (3B でも届かない) |

### elt-lm の **構造的アドバンテージ**

1. **L-iteration = per-token の「think longer」** — o1 / R1 系と motivation 一致。
   「次に何呼ぶか」の判断は CoT ヘビーなので L=4 が効く。
2. **GRPO `correct × format` verifier** — ツールコール JSON の schema 準拠を直接報酬化。
   1B 級最大の失敗モード (フォーマット破綻) に最適な訓練信号。
3. **Python-exec verifier (task 29-30) 実装済み** — code-as-tool 評価がフレームワーク
   として既に乗っている。
4. **task 28, 31 でエージェント系・汎用 AI ソース 5 件取込済み** — corpus 下地あり。

### スケール限界 (正直に)

- ツール選択の世界知識は weight-bound → L=4 で補えない
- 5 step 以上の horizon は 1B では破綻率が跳ね上がる
- IFEval 上限は 40–55 付近 (SmolLM2-1.7B が 56.7 で参考上限)

### 勝てる戦場

**「L-elastic tool caller」というポジション**:
- L=1: < 50ms の軽量ルーティング (intent classification, simple func call)
- L=4: 複雑 JSON schema や 2–3 hop ReAct
- **同一 ckpt で速度/品質を runtime 切替** — エッジエージェント / latency-sensitive な
  agentic UI (音声 AI の tool router, IDE 内 copilot の一次判断) に直刺し

## 4. コーディングエージェント

### 同規模ベースライン

| モデル | HumanEval | MBPP | 備考 |
|---|---|---|---|
| Qwen2.5-Coder-1.5B | **43.9** | 69.2 | 5.2T code tokens |
| DeepSeek-Coder-1.3B | 34.8 | 55.6 | 2T code tokens |
| CodeGemma-2B | 31.1 | 43.6 | — |
| StarCoder2-3B | 31.7 | — | — |

### elt-lm-base-1.5B の予測

| タスク層 | 代表ベンチ | 現実予測 |
|---|---|---|
| 単発関数生成 | HumanEval | 20–35% |
| 手続き的問題 | MBPP | 30–50% |
| 競プロ系 | LiveCodeBench Easy | **15–30% (L=4 + GRPO で伸びしろ)** |
| FIM / 補完 | HumanEvalFIM | **非対応** (FIM objective 未実装) |
| multi-file edit | Aider bench | 届かない (3B+ 必要) |
| SWE-bench Verified | — | ~0% (7B+ with scaffolding が下限) |

### elt-lm が **コード領域で強い構造的理由**

1. **Python-exec verifier は DeepSeek-R1 / Coder-RL と同じ学習信号**
   「コンパイル通る × テスト pass × フォーマット」を GRPO 報酬に直結できる。
   **コードは RL と相性が最も良いドメイン** (報酬が曖昧でない)。
2. **L-iteration と algorithmic reasoning の整合**
   Universal Transformer 系は算術・パターンマッチ・木構造走査で固定パラメータ比の
   改善を実証済み。HumanEval 難問 / 競プロ系はこの層で L=4 が効く。
3. **task 23 (コード/ツール/数学 ソース) が既に corpus に入っている**
   下地は存在する。

### 弱点 (正直に)

- **コード専用 token 比率の不足**: DeepSeek-Coder-1.3B は 2T code、
  Qwen2.5-Coder-1.5B は 5.2T code。elt-lm の code 比率が 10–20% なら 0.5–2B tokens ──
  **1 桁以上のギャップ**。HumanEval 40% 台には届きにくい。
- **FIM objective 未導入**: Copilot 的補完は causal-only では弱い。
- **長コンテキスト**: repo-level 編集は 8k–32k 必要。現行 `max_seq_len` は smoke=16、
  本番でも Sliding/YaRN 等の拡張が要る。
- **ライブラリ API 知識は weight-bound** → L=4 で補えない。

### 勝てる戦場

- **Exec-verified RL で小ドメイン特化**: Python 単関数 / LeetCode / Project Euler レベル。
  **HumanEval+ 30–40%、小算法タスクで 3B モデルに接近し得る**。
- **L-elastic code router**: IDE 統合で L=1 補完 / L=4 「もう一度考える」ボタン。
  **同一 ckpt で両立するのは現状の OSS でユニーク**。
- **教育・競プロ支援**: 検証可能な小問題に絞れば 1B でも用途あり。

## 5. 総合判断 — elt-lm の contribution は何か

**ベンチ絶対値で同規模 SOTA を抜きに行く設計ではない** (Qwen2.5-Coder 1.5B の 5.2T code
token 相当を追う時間・予算がない) という前提で、以下 3 点に価値が集中する:

### (a) L-elastic inference (同 ckpt で L=1..4 を runtime 切替)
- 1B 級 OSS で唯一。
- 速度/品質の Pareto frontier を単一チェックポイントで提供。
- エッジ・IDE・音声など **latency-sensitive な用途に直刺し**。

### (b) Exec-verified GRPO を code で最大活用
- Python-exec verifier (task 29-30) + GRPO = DeepSeek-R1 / Coder-RL 直系の学習信号。
- **コードこそ本プロジェクトの RL が最も効く領域**。

### (c) ILSD + L-curriculum の faithful な再現
- 論文 (arXiv:2604.09168) の PyTorch port として存在意義。
- 再現性・読解可能性 (paper equations preserved verbatim) が研究者向けの一次価値。

## 6. 推奨アクション (優先度順)

1. **`inference_sweep` に HumanEval / GSM8K / MMLU-STEM / LiveCodeBench を追加**
   → dashboard の Inference Pareto パネルがそのまま L-sweep 結果を表示できる。
   論文 figure 1 候補。
2. **GRPO Phase-3 で python-exec verifier を GSM8K と並列に回す**
   → 数学より報酬設計が安定。HumanEval+ への早期シグナル取得。
3. **Phase-1 の code 比率を corpus manifest で明示化**
   → 15% → 25–30% に寄せれば HumanEval +5–10pt は現実的。task 32 DL-2 の選択基準に反映。
4. **FIM objective の追加** (新規 Phase 1.5)
   → `<fim_prefix>/<fim_suffix>/<fim_middle>` token 追加 + loss mixing 10%。
   補完ベンチ (HumanEvalFIM) を開けるようになる。
5. **BFCL Simple に絞った instruct SFT を Phase 2 に追加**
   → xLAM-1B レベルの単発 tool call を狙う。GRPO format reward と噛み合う。

## 7. 非目的 (本プロジェクトで追わない)

- **7B+ スケールでの SOTA 競争** — 計算予算・データ量で prohibitive。
- **GAIA / SWE-bench Verified の汎用エージェント** — 1B の土俵外。
- **翻訳・多言語ベンチ** — corpus が英中日偏重で不利、かつ L-iteration の強みが出ない。
- **Multimodal** — 本論文スコープ外、別プロジェクト相当。

## 参考 (ベンチ情報源)

- [Qwen2.5-LLM blog](https://qwenlm.github.io/blog/qwen2.5-llm/)
- [Qwen2.5 Technical Report (arXiv:2412.15115)](https://arxiv.org/pdf/2412.15115)
- [Qwen2.5-Coder Technical Report (arXiv:2409.12186)](https://arxiv.org/pdf/2409.12186)
- [Llama 3.2 benchmark insights](https://amdadulhaquemilon.medium.com/llama-3-2-benchmark-insights-and-revolutionizing-edge-ai-and-vision-88542fe3dc0d)
- [SmolLM2 paper notes](https://ritvik19.medium.com/papers-explained-310-smollm2-53991a485d7b)
- [Chinchilla scaling laws (arXiv:2203.15556)](https://arxiv.org/abs/2203.15556)
- [Berkeley Function Calling Leaderboard (BFCL)](https://gorilla.cs.berkeley.edu/leaderboard.html)
- [Small LM leaderboard](https://awesomeagents.ai/leaderboards/small-language-model-leaderboard/)

## 次セッションへの引き継ぎ

- この分析は 2026-04-21 時点の公開ベンチを元にした**予測**であり、実測ではない。
  Phase-1 pretrain 完了後に `inference_sweep` で実測が出た段階で**数値を上書き**すること。
- 推奨アクション 1 (inference_sweep 拡張) は Phase D の自然な続編で、実装コスト低・
  論文価値高なので**次の実装タスク候補として筆頭**。
- 推奨アクション 4 (FIM objective) は Phase 1.5 として新規タスク化が必要。
  pretrain 再走が要るので、task 32 (DL-2) 完了 → bin 再生成 → FIM 対応 pretrain の順。
