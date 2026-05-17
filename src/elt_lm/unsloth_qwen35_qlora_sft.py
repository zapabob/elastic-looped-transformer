"""Unsloth QLoRA SFT launcher for the local Huihui Qwen3.5-4B cache.

The module keeps Unsloth imports inside the training path so dry-run dataset
checks and unit tests do not require the optional fine-tuning stack.
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import statistics
from dataclasses import dataclass
from glob import glob
from pathlib import Path
from typing import Any, Iterable


DEFAULT_MODEL_CACHE = (
    "H:/hf_cache/hub/models--huihui-ai--"
    "Huihui-Qwen3.5-4B-Claude-4.6-Opus-abliterated"
)
DEFAULT_HF_DATASET = "Aratako/Synthetic-Japanese-Roleplay-NSFW-Claude-4.5s-3.5k-formatted"
DEFAULT_OUTPUT_DIR = "H:/elt_data/runs/huihui_qwen35_4b_roleplay_unsloth_qlora"
DEFAULT_TOKENIZER_PATH = "H:/Qwen3.5-9B-official-hf"
DEFAULT_CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "<|im_start|>{{ message['role'] }}\n"
    "{{ message['content'] }}<|im_end|>\n"
    "{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)

ROLE_MAP = {
    "human": "user",
    "user": "user",
    "assistant": "assistant",
    "gpt": "assistant",
    "bot": "assistant",
    "system": "system",
}

# Conservative lexical guardrails for the default training path. The dataset is
# still adult-oriented; this filter only removes obvious disallowed/high-risk
# records before SFT. It intentionally avoids printing matched text.
ADULT_CONSENT_BLOCKLIST = (
    "未成年",
    "児童",
    "小学生",
    "中学生",
    "高校生",
    "幼女",
    "幼い",
    "17歳",
    "16歳",
    "15歳",
    "14歳",
    "13歳",
    "12歳",
    "拒否",
    "抵抗",
    "無理やり",
    "強制",
    "脅迫",
    "拘束",
    "監禁",
    "拷問",
    "レイプ",
    "強姦",
    "近親",
)
MOJIBAKE_MARKERS = ("縺", "繧", "荳", "譁", "蜿", "螟", "逧", "莠", "�")


ADULT_CONSENT_BLOCKLIST_EXTRA = (
    "未成年",
    "児童",
    "小学生",
    "中学生",
    "高校生",
    "幼女",
    "幼い",
    "ロリ",
    "JS",
    "JC",
    "JK",
    "12歳",
    "13歳",
    "14歳",
    "15歳",
    "16歳",
    "17歳",
    "underage",
    "minor",
    "child",
    "rape",
    "レイプ",
    "強姦",
    "強制",
    "無理やり",
    "抵抗",
    "監禁",
    "誘拐",
    "薬物",
    "覚醒剤",
    "メタンフェタミン",
    "シャブ",
    "ドラッグ",
    "キメセク",
)


@dataclass(frozen=True)
class PreparedDataset:
    dataset: Any
    rows_before: int
    rows_after_filter: int
    source_stats: list[dict[str, Any]]


def resolve_model_path(model_path: str | Path) -> Path:
    """Resolve a HF cache root or direct snapshot directory to loadable files."""

    path = Path(model_path)
    snapshots = path / "snapshots"
    if not snapshots.is_dir():
        return path

    ref_main = path / "refs" / "main"
    if ref_main.is_file():
        revision = ref_main.read_text(encoding="utf-8").strip()
        candidate = snapshots / revision
        if candidate.is_dir():
            return candidate

    candidates = [p for p in snapshots.iterdir() if p.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"no snapshots found under {snapshots}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def validate_jsonl_file(path: str | Path) -> None:
    """Fail early if a local file is not true one-object-per-line JSONL."""

    jsonl = Path(path)
    with jsonl.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{jsonl}:{line_no} is not valid JSONL "
                    f"({exc.msg} at column {exc.colno}). "
                    "Repair embedded raw newlines or export one JSON object per line."
                ) from exc


def resolve_parquet_data_files(spec: str | Path) -> str | list[str]:
    """Resolve one parquet path, a glob, or a semicolon/comma-separated list."""

    parts = [
        part.strip()
        for chunk in str(spec).split(";")
        for part in chunk.split(",")
        if part.strip()
    ]
    resolved: list[str] = []
    for part in parts:
        matches = sorted(glob(part))
        resolved.extend(matches or [str(Path(part))])
    if len(resolved) == 1:
        return resolved[0]
    return resolved


def _source_specs(args: argparse.Namespace) -> list[tuple[str, str, str | Path]]:
    specs: list[tuple[str, str, str | Path]] = []
    if args.jsonl:
        specs.append(("jsonl", str(args.jsonl), args.jsonl))
    elif args.parquet:
        specs.append(("parquet", str(args.parquet), args.parquet))
    else:
        specs.append(("hf", str(args.hf_dataset), args.hf_dataset))

    for extra_jsonl in args.extra_jsonl:
        specs.append(("jsonl", str(extra_jsonl), extra_jsonl))
    for extra_parquet in args.extra_parquet:
        specs.append(("parquet", str(extra_parquet), extra_parquet))
    return specs


def _content_from_turn(turn: dict[str, Any]) -> str:
    content = turn.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts)
    return ""


def coerce_messages(example: dict[str, Any]) -> list[dict[str, str]] | None:
    """Normalize common chat schemas to HF chat-template messages."""

    raw_messages = example.get("messages")
    if isinstance(raw_messages, list):
        messages: list[dict[str, str]] = []
        for turn in raw_messages:
            if not isinstance(turn, dict):
                continue
            role = ROLE_MAP.get(str(turn.get("role", "")).lower())
            content = _content_from_turn(turn).strip()
            if role and content:
                messages.append({"role": role, "content": content})
        return messages or None

    conversations = example.get("conversations")
    if isinstance(conversations, list):
        converted: list[dict[str, str]] = []
        for turn in conversations:
            if not isinstance(turn, dict):
                continue
            role = ROLE_MAP.get(str(turn.get("from", turn.get("role", ""))).lower())
            value = turn.get("value", turn.get("content", ""))
            content = value.strip() if isinstance(value, str) else ""
            if role and content:
                converted.append({"role": role, "content": content})
        return converted or None

    system = example.get("system")
    user = example.get("user", example.get("prompt"))
    assistant = example.get("assistant", example.get("response", example.get("completion")))
    messages = []
    if isinstance(system, str) and system.strip():
        messages.append({"role": "system", "content": system.strip()})
    if isinstance(user, str) and user.strip():
        messages.append({"role": "user", "content": user.strip()})
    if isinstance(assistant, str) and assistant.strip():
        messages.append({"role": "assistant", "content": assistant.strip()})
    if any(m["role"] == "assistant" for m in messages):
        return messages
    return None


def record_text_for_filter(example: dict[str, Any]) -> str:
    text = example.get("text")
    if isinstance(text, str):
        return text

    messages = coerce_messages(example)
    if messages:
        return "\n".join(turn["content"] for turn in messages)
    return ""


def passes_adult_consent_filter(example: dict[str, Any]) -> bool:
    text = record_text_for_filter(example)
    if any(term in text for term in ADULT_CONSENT_BLOCKLIST):
        return False
    lowered = text.lower()
    return not any(term.lower() in lowered for term in ADULT_CONSENT_BLOCKLIST_EXTRA)


def looks_like_mojibake(text: str) -> bool:
    marker_hits = sum(text.count(marker) for marker in MOJIBAKE_MARKERS)
    return marker_hits >= 3


def format_training_text(example: dict[str, Any], tokenizer: Any) -> str:
    text = example.get("text")
    if isinstance(text, str) and text.strip():
        return text.strip()

    messages = coerce_messages(example)
    if not messages:
        raise ValueError("example does not contain messages/text or prompt-response fields")

    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    if not isinstance(rendered, str) or not rendered.strip():
        raise ValueError("tokenizer chat template produced empty text")
    return rendered


def ensure_chat_template(tokenizer: Any) -> None:
    if not getattr(tokenizer, "chat_template", None):
        tokenizer.chat_template = DEFAULT_CHAT_TEMPLATE
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token


def _load_raw_dataset(args: argparse.Namespace, kind: str | None = None, spec: str | Path | None = None) -> Any:
    from datasets import load_dataset

    token = args.hf_token or os.environ.get("HF_TOKEN")
    kind = kind or ("jsonl" if args.jsonl else "parquet" if args.parquet else "hf")
    spec = spec or args.jsonl or args.parquet or args.hf_dataset
    if kind == "jsonl":
        validate_jsonl_file(spec)
        return load_dataset("json", data_files=str(Path(spec)), split="train")
    if kind == "parquet":
        return load_dataset("parquet", data_files=resolve_parquet_data_files(spec), split="train")
    return load_dataset(str(spec), args.hf_config, split=args.hf_split, token=token)


def prepare_dataset(args: argparse.Namespace, tokenizer: Any) -> PreparedDataset:
    def render(example: dict[str, Any]) -> dict[str, str]:
        return {"text": format_training_text(example, tokenizer)}

    from datasets import concatenate_datasets

    rendered_parts: list[Any] = []
    source_stats: list[dict[str, Any]] = []
    rows_before_total = 0
    rows_after_total = 0

    for kind, label, spec in _source_specs(args):
        raw = _load_raw_dataset(args, kind, spec)
        rows_before = len(raw)
        dataset = raw

        if args.safety_filter == "adult-consent":
            dataset = dataset.filter(
                passes_adult_consent_filter,
                desc=f"adult-consent filter: {label}",
            )
        rows_after_filter = len(dataset)
        rows_before_total += rows_before
        rows_after_total += rows_after_filter
        source_stats.append({
            "source": label,
            "kind": kind,
            "rows_before": rows_before,
            "rows_after_filter": rows_after_filter,
        })
        if rows_after_filter == 0:
            continue

        rendered = dataset.map(
            render,
            remove_columns=dataset.column_names,
            desc=f"render chat template: {label}",
        )
        token_length_limit = args.max_train_token_length or args.max_seq_length
        if args.drop_over_max_seq_length or args.max_train_token_length:
            rendered = rendered.filter(
                lambda example: len(
                    tokenizer(example["text"], add_special_tokens=False).input_ids
                ) <= token_length_limit,
                desc=f"drop over token length {token_length_limit}: {label}",
            )
        source_stats[-1]["rows_after_length_filter"] = len(rendered)
        rendered_parts.append(rendered)

    if not rendered_parts:
        raise ValueError("dataset is empty after filtering")

    dataset = rendered_parts[0] if len(rendered_parts) == 1 else concatenate_datasets(rendered_parts)
    return PreparedDataset(
        dataset=dataset,
        rows_before=rows_before_total,
        rows_after_filter=rows_after_total,
        source_stats=source_stats,
    )


def _percentile(values: list[int], pct: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * pct)))
    return int(ordered[idx])


def dry_run(args: argparse.Namespace) -> None:
    from transformers import AutoTokenizer

    model_path = resolve_model_path(args.model_path)
    tokenizer_path = resolve_model_path(args.tokenizer_path or args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), trust_remote_code=True)
    ensure_chat_template(tokenizer)
    prepared = prepare_dataset(args, tokenizer)
    if args.dry_run_rows <= 0:
        sample_count = len(prepared.dataset)
    else:
        sample_count = min(args.dry_run_rows, len(prepared.dataset))
    token_lengths: list[int] = []
    mojibake_suspect = 0
    for row in prepared.dataset.select(range(sample_count)):
        mojibake_suspect += int(looks_like_mojibake(row["text"]))
        token_lengths.append(len(tokenizer(row["text"], add_special_tokens=False).input_ids))

    summary = {
        "mode": "dry_run",
        "model_path": str(model_path),
        "tokenizer_path": str(tokenizer_path),
        "source": dataset_source_label(args),
        "rows_before": prepared.rows_before,
        "rows_after_filter": prepared.rows_after_filter,
        "rows_after_length_filter": len(prepared.dataset),
        "source_stats": prepared.source_stats,
        "checked_rows": sample_count,
        "safety_filter": args.safety_filter,
        "max_seq_length": args.max_seq_length,
        "token_length_min": min(token_lengths) if token_lengths else 0,
        "token_length_mean": round(statistics.fmean(token_lengths), 2) if token_lengths else 0,
        "token_length_p50": int(statistics.median(token_lengths)) if token_lengths else 0,
        "token_length_p90": _percentile(token_lengths, 0.90),
        "token_length_p95": _percentile(token_lengths, 0.95),
        "token_length_p99": _percentile(token_lengths, 0.99),
        "token_length_max": max(token_lengths) if token_lengths else 0,
        "over_max_seq_length": sum(length > args.max_seq_length for length in token_lengths),
        "mojibake_suspect_rows": mojibake_suspect,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def _make_sft_config(sft_config_cls: type, kwargs: dict[str, Any]) -> Any:
    signature = inspect.signature(sft_config_cls.__init__)
    accepted = set(signature.parameters)

    if "max_seq_length" in accepted:
        kwargs["max_seq_length"] = kwargs.pop("max_length")
    elif "max_length" not in accepted:
        kwargs.pop("max_length", None)

    filtered = {key: value for key, value in kwargs.items() if key in accepted}
    return sft_config_cls(**filtered)


def _make_sft_trainer(trainer_cls: type, tokenizer: Any, kwargs: dict[str, Any]) -> Any:
    signature = inspect.signature(trainer_cls.__init__)
    accepted = set(signature.parameters)
    if "tokenizer" in accepted:
        kwargs["tokenizer"] = tokenizer
    elif "processing_class" in accepted:
        kwargs["processing_class"] = tokenizer
    return trainer_cls(**{key: value for key, value in kwargs.items() if key in accepted})


def train(args: argparse.Namespace) -> None:
    if not args.jsonl and not args.accept_dataset_terms:
        raise SystemExit(
            "Refusing to train from the HF dataset/parquet without --accept-dataset-terms. "
            "The dataset card lists CC-BY-NC-SA 4.0 and an Anthropic terms note; "
            "confirm your intended local use is permitted before launching training."
        )
    if args.disable_torch_compile:
        # Windows triton-windows can hit tcc/Inductor compile failures even
        # after CUDA training itself is healthy. Disable Dynamo/Inductor before
        # Unsloth imports so the run uses eager CUDA kernels instead.
        os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
        os.environ.setdefault("TORCHINDUCTOR_DISABLE", "1")

    try:
        from unsloth import FastLanguageModel, is_bfloat16_supported
        from unsloth.chat_templates import train_on_responses_only
        from trl import SFTConfig, SFTTrainer
    except ImportError as exc:
        raise SystemExit(
            "Unsloth training deps are missing. Install in this environment with: "
            "uv pip install unsloth trl peft accelerate bitsandbytes"
        ) from exc
    import torch

    model_path = resolve_model_path(args.model_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dtype = None if args.dtype == "auto" else {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[args.dtype]
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(model_path),
        max_seq_length=args.max_seq_length,
        dtype=dtype,
        load_in_4bit=True,
        token=args.hf_token or os.environ.get("HF_TOKEN"),
    )
    if args.tokenizer_path:
        # The local cache snapshot can contain weights only; use the full Qwen
        # tokenizer directory when provided.
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            str(resolve_model_path(args.tokenizer_path)),
            trust_remote_code=True,
        )
    ensure_chat_template(tokenizer)

    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_rank,
        target_modules=[part.strip() for part in args.target_modules.split(",") if part.strip()],
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth" if args.gradient_checkpointing else False,
        random_state=args.seed,
        use_rslora=args.use_rslora,
        loftq_config=None,
    )

    prepared = prepare_dataset(args, tokenizer)
    train_dataset = prepared.dataset
    eval_dataset = None
    if args.eval_ratio > 0:
        split = train_dataset.train_test_split(test_size=args.eval_ratio, seed=args.seed)
        train_dataset = split["train"]
        eval_dataset = split["test"]

    bf16 = bool(is_bfloat16_supported())
    config_kwargs = {
        "output_dir": str(output_dir),
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "warmup_ratio": args.warmup_ratio,
        "max_steps": args.max_steps,
        "num_train_epochs": args.num_train_epochs,
        "learning_rate": args.learning_rate,
        "fp16": not bf16,
        "bf16": bf16,
        "logging_steps": args.logging_steps,
        "eval_strategy": "steps" if eval_dataset is not None else "no",
        "evaluation_strategy": "steps" if eval_dataset is not None else "no",
        "eval_steps": args.eval_steps,
        "optim": args.optim,
        "weight_decay": args.weight_decay,
        "lr_scheduler_type": args.lr_scheduler_type,
        "seed": args.seed,
        "save_steps": args.save_steps,
        "save_total_limit": args.save_total_limit,
        "report_to": "none" if args.report_to == "none" else args.report_to.split(","),
        # Keep this as None on Windows. Unsloth's compiled SFTTrainer class can
        # fail to pickle across a SpawnPoolWorker even when num_proc=1.
        "dataset_num_proc": None,
        "dataset_text_field": "text",
        "packing": args.packing,
        "max_length": args.max_seq_length,
    }
    training_args = _make_sft_config(SFTConfig, config_kwargs)
    trainer = _make_sft_trainer(
        SFTTrainer,
        tokenizer,
        {
            "model": model,
            "train_dataset": train_dataset,
            "eval_dataset": eval_dataset,
            "args": training_args,
        },
    )

    if args.train_on_responses_only:
        trainer = train_on_responses_only(
            trainer,
            instruction_part=args.instruction_part,
            response_part=args.response_part,
        )

    manifest = {
        "model_path": str(model_path),
        "source": dataset_source_label(args),
        "rows_before": prepared.rows_before,
        "rows_after_filter": prepared.rows_after_filter,
        "rows_after_length_filter": len(prepared.dataset),
        "source_stats": prepared.source_stats,
        "train_rows": len(train_dataset),
        "eval_rows": len(eval_dataset) if eval_dataset is not None else 0,
        "safety_filter": args.safety_filter,
        "max_seq_length": args.max_seq_length,
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "learning_rate": args.learning_rate,
        "max_steps": args.max_steps,
        "num_train_epochs": args.num_train_epochs,
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint or None)
    adapter_dir = output_dir / "adapter"
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    if eval_dataset is not None:
        eval_metrics = latest_eval_metrics_from_state(trainer.state.log_history)
        if eval_metrics is None:
            eval_metrics = trainer.evaluate()
        (output_dir / "eval_metrics.json").write_text(
            json.dumps(eval_metrics, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    if args.save_merged_16bit:
        merged_dir = output_dir / "merged_16bit"
        model.save_pretrained_merged(str(merged_dir), tokenizer, save_method="merged_16bit")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--hf-dataset", default=DEFAULT_HF_DATASET)
    source.add_argument("--jsonl", help="Local JSONL file with messages/text or prompt-response fields.")
    source.add_argument("--parquet", help="Local parquet file, useful when Python SSL blocks Hub downloads.")
    parser.add_argument(
        "--extra-jsonl",
        action="append",
        default=[],
        help="Additional local JSONL file to filter, render, and mix with the primary source.",
    )
    parser.add_argument(
        "--extra-parquet",
        action="append",
        default=[],
        help="Additional parquet file/glob to filter, render, and mix with the primary source.",
    )
    parser.add_argument("--hf-config", default="default")
    parser.add_argument("--hf-split", default="train")
    parser.add_argument("--hf-token", default=None)
    parser.add_argument("--accept-dataset-terms", action="store_true")

    parser.add_argument("--model-path", default=DEFAULT_MODEL_CACHE)
    parser.add_argument("--tokenizer-path", default=DEFAULT_TOKENIZER_PATH)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--safety-filter", choices=["adult-consent"], default="adult-consent")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--dry-run-rows", type=int, default=32)

    parser.add_argument("--max-seq-length", type=int, default=4096)
    parser.add_argument(
        "--drop-over-max-seq-length",
        action="store_true",
        help="Drop rendered rows longer than max_seq_length instead of truncating them during SFT.",
    )
    parser.add_argument(
        "--max-train-token-length",
        type=int,
        default=None,
        help="Optional stricter rendered-token length limit while keeping max_seq_length unchanged.",
    )
    parser.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"], default="auto")
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument(
        "--target-modules",
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    )
    parser.add_argument("--use-rslora", action="store_true")
    parser.add_argument("--no-gradient-checkpointing", dest="gradient_checkpointing", action="store_false")
    parser.set_defaults(gradient_checkpointing=True)

    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--lr-scheduler-type", default="cosine")
    parser.add_argument("--optim", default="adamw_8bit")
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--eval-ratio", type=float, default=0.03)
    parser.add_argument("--eval-steps", type=int, default=100)
    parser.add_argument("--logging-steps", type=int, default=5)
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument("--save-total-limit", type=int, default=3)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--report-to", default="none")
    parser.add_argument("--packing", action="store_true")
    parser.add_argument("--resume-from-checkpoint", default=None)
    parser.add_argument("--save-merged-16bit", action="store_true")
    parser.add_argument("--enable-torch-compile", dest="disable_torch_compile", action="store_false")
    parser.set_defaults(disable_torch_compile=True)

    parser.add_argument("--train-on-responses-only", action="store_true", default=True)
    parser.add_argument("--instruction-part", default="<|im_start|>user\n")
    parser.add_argument("--response-part", default="<|im_start|>assistant\n")
    return parser


def cli(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.dry_run:
        dry_run(args)
    else:
        train(args)


def dataset_source_label(args: argparse.Namespace) -> str:
    return ";".join(label for _, label, _ in _source_specs(args))


def latest_eval_metrics_from_state(log_history: Iterable[dict[str, object]]) -> dict[str, object] | None:
    """Return the latest eval metrics already recorded by Trainer, if any."""
    for row in reversed(list(log_history)):
        if "eval_loss" not in row:
            continue
        metrics = {key: value for key, value in row.items() if key.startswith("eval_")}
        if "step" in row:
            metrics["step"] = row["step"]
        if "epoch" in row:
            metrics["epoch"] = row["epoch"]
        return metrics
    return None


if __name__ == "__main__":
    cli()
