from __future__ import annotations

import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def parse_size(text: str) -> int:
    value = text.strip().upper()
    units = {
        "B": 1,
        "KB": 1024,
        "MB": 1024**2,
        "GB": 1024**3,
    }
    for suffix, scale in sorted(units.items(), key=lambda item: len(item[0]), reverse=True):
        if value.endswith(suffix):
            return int(float(value[: -len(suffix)]) * scale)
    return int(value)


def copy_model_files(base_dir: Path, tokenizer_dir: Path, out_dir: Path) -> None:
    skip_suffixes = (".safetensors", ".bin")
    skip_names = {"model.safetensors.index.json", "pytorch_model.bin.index.json"}
    for src in base_dir.iterdir():
        if src.is_dir() or src.name in skip_names or src.name.endswith(skip_suffixes):
            continue
        shutil.copy2(src, out_dir / src.name)

    for name in (
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "vocab.json",
        "merges.txt",
        "special_tokens_map.json",
    ):
        src = tokenizer_dir / name
        if src.exists():
            shutil.copy2(src, out_dir / src.name)


def load_lora(adapter_dir: Path) -> tuple[dict[str, tuple[torch.Tensor, torch.Tensor]], float]:
    with (adapter_dir / "adapter_config.json").open("r", encoding="utf-8") as fh:
        config = json.load(fh)

    scale = float(config["lora_alpha"]) / float(config["r"])
    adapter_path = adapter_dir / "adapter_model.safetensors"
    tensors: dict[str, torch.Tensor] = {}
    with safe_open(adapter_path, framework="pt", device="cpu") as fh:
        for key in fh.keys():
            tensors[key] = fh.get_tensor(key)

    updates: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    prefix = "base_model.model."
    for key, tensor in tensors.items():
        if not key.endswith(".lora_A.weight"):
            continue
        stem = key[: -len(".lora_A.weight")]
        b_key = f"{stem}.lora_B.weight"
        if b_key not in tensors:
            raise KeyError(f"missing LoRA B tensor for {key}")
        base_name = stem
        if base_name.startswith(prefix):
            base_name = base_name[len(prefix) :]
        updates[f"{base_name}.weight"] = (tensor, tensors[b_key])
    return updates, scale


def tensor_nbytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def save_group(
    out_dir: Path,
    groups: list[Path],
    weight_map: dict[str, str],
    tensors: dict[str, torch.Tensor],
) -> None:
    if not tensors:
        return
    part = out_dir / f"model-part-{len(groups) + 1:05d}.safetensors"
    save_file(tensors, part, metadata={"format": "pt"})
    for name in tensors:
        weight_map[name] = part.name
    groups.append(part)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge a LoRA adapter directly into sharded safetensors."
    )
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--max-shard-size", default="1536MB")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    base_dir = Path(args.base_model)
    adapter_dir = Path(args.adapter)
    tokenizer_dir = Path(args.tokenizer) if args.tokenizer else adapter_dir
    out_dir = Path(args.out_dir)
    max_shard_size = parse_size(args.max_shard_size)

    if out_dir.exists():
        if not args.force:
            raise SystemExit(f"{out_dir} already exists; pass --force to replace it")
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    copy_model_files(base_dir, tokenizer_dir, out_dir)
    updates, scale = load_lora(adapter_dir)

    with (base_dir / "model.safetensors.index.json").open("r", encoding="utf-8") as fh:
        base_index = json.load(fh)

    names_by_file: dict[str, list[str]] = defaultdict(list)
    for name, shard in base_index["weight_map"].items():
        names_by_file[shard].append(name)

    groups: list[Path] = []
    output_weight_map: dict[str, str] = {}
    current: dict[str, torch.Tensor] = {}
    current_size = 0
    total_size = 0
    merged_count = 0

    for shard_name in sorted(names_by_file):
        print(f"reading {shard_name}", flush=True)
        with safe_open(base_dir / shard_name, framework="pt", device="cpu") as fh:
            for name in names_by_file[shard_name]:
                tensor = fh.get_tensor(name)
                if name in updates:
                    lora_a, lora_b = updates[name]
                    update = torch.matmul(lora_b.float(), lora_a.float()).mul_(scale)
                    tensor = tensor.float().add_(update).to(dtype=tensor.dtype)
                    merged_count += 1
                tensor = tensor.contiguous()
                size = tensor_nbytes(tensor)
                if current and current_size + size > max_shard_size:
                    save_group(out_dir, groups, output_weight_map, current)
                    current = {}
                    current_size = 0
                current[name] = tensor
                current_size += size
                total_size += size
        print(f"processed {shard_name}", flush=True)

    save_group(out_dir, groups, output_weight_map, current)

    final_weight_map: dict[str, str] = {}
    total_parts = len(groups)
    for index, part in enumerate(groups, start=1):
        final_name = f"model-{index:05d}-of-{total_parts:05d}.safetensors"
        final_path = out_dir / final_name
        part.rename(final_path)
        for tensor_name, part_name in output_weight_map.items():
            if part_name == part.name:
                final_weight_map[tensor_name] = final_name

    with (out_dir / "model.safetensors.index.json").open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "metadata": {"total_size": total_size},
                "weight_map": final_weight_map,
            },
            fh,
            indent=2,
            sort_keys=True,
        )
        fh.write("\n")

    missing = sorted(set(updates) - set(base_index["weight_map"]))
    print(f"merged_lora_tensors={merged_count}", flush=True)
    print(f"missing_lora_targets={len(missing)}", flush=True)
    if missing:
        for name in missing:
            print(f"missing: {name}", flush=True)


if __name__ == "__main__":
    main()
