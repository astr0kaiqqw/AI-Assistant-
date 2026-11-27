"""Score the pinned Tiny Shakespeare validation split and save fixed-prompt samples."""

import argparse
import csv
import hashlib
import json
import math
import platform
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import torch
import torch.nn.functional as F

from mini_llm.data import DATASET_SHA256, DEFAULT_DATA, PACKAGE_DIR, dataset_splits, load_dataset
from mini_llm.generate import generate_tokens, load_checkpoint
from mini_llm.runtime import select_device

EVALUATIONS = PACKAGE_DIR / "evaluations"
PROMPTS = ("First Citizen:", "ROMEO:", "To be, or not to be")


def new_result_dir(root, label):
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    path = root / f"{stamp}-{label}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def write_json(path, value):
    with path.open("x", encoding="utf-8") as file:
        json.dump(value, file, indent=2, ensure_ascii=False, allow_nan=False)
        file.write("\n")


def file_sha256(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def quality_metrics(nll, tokens, byte_count):
    loss = nll / tokens
    return {
        "negative_log_likelihood_nats": nll,
        "validation_loss": loss,
        "perplexity": math.exp(loss),
        "bits_per_byte": nll / (math.log(2) * byte_count),
        "evaluated_tokens": tokens,
        "evaluated_bytes": byte_count,
    }


@torch.inference_mode()
def score_validation(model, validation, tokenizer, context, batch_size, device):
    """Score each target exactly once using contiguous, non-overlapping target blocks."""
    if not 1 <= context <= model.config.context_length or batch_size < 1 or len(validation) < 2:
        raise ValueError("Invalid scoring context, batch size or validation length")
    model.eval()
    target_count = len(validation) - 1
    full_count, remainder = divmod(target_count, context)
    losses = []
    for first in range(0, full_count, batch_size):
        block_count = min(batch_size, full_count - first)
        start = first * context
        end = start + block_count * context
        inputs = validation[start:end].view(-1, context).to(device)
        targets = validation[start + 1:end + 1].to(device)
        logits, _ = model(inputs)
        losses.append(F.cross_entropy(logits.flatten(0, 1), targets, reduction="sum").item())

    if remainder:
        start = full_count * context
        logits, _ = model(validation[start:-1].unsqueeze(0).to(device))
        losses.append(F.cross_entropy(logits[0], validation[start + 1:].to(device), reduction="sum").item())

    # This scoring protocol requires a lossless character tokenizer. The first
    # validation character supplies context only and contributes no bytes.
    byte_count = len(tokenizer.decode(validation[1:].tolist()).encode("utf-8"))
    return quality_metrics(math.fsum(losses), target_count, byte_count)


def training_metadata(path, step, config, split):
    config_path = path.with_name("config.json")
    log_path = path.with_name("metrics.csv")
    if not config_path.is_file() or not log_path.is_file():
        return None
    run = json.loads(config_path.read_text(encoding="utf-8"))
    if run.get("model") != config or run.get("dataset", {}).get("sha256") != split["sha256"]:
        return None
    with log_path.open(newline="", encoding="utf-8") as file:
        rows = [row for row in csv.DictReader(file) if int(row["step"]) <= step]
    if [int(row["step"]) for row in rows] != list(range(1, step + 1)) or not rows:
        return None
    return {
        "logged_step_seconds_total": math.fsum(float(row["step_seconds"]) for row in rows),
        "timing_source": str(log_path),
        "sampled_validation_loss_at_checkpoint": float(rows[-1]["validation_loss"])
        if rows[-1]["validation_loss"] else None,
        "device": run.get("device"),
    }


def evaluate(args):
    device = select_device(args.device)
    torch.set_num_threads(args.cpu_threads)
    torch.manual_seed(args.seed)
    path = args.checkpoint.resolve(strict=True)
    checkpoint_sha = file_sha256(path)
    model, tokenizer, step = load_checkpoint(path, device)
    # The evaluation CLI uses an existing, checksum-verified dataset cache.
    data_path = args.data.resolve(strict=True)
    text = load_dataset(data_path)
    fitted, training, validation = dataset_splits(text)
    if tokenizer.vocabulary != fitted.vocabulary:
        raise ValueError("Checkpoint vocabulary differs from the pinned training-split vocabulary")
    split = {
        "name": "Tiny Shakespeare",
        "path": str(data_path),
        "sha256": DATASET_SHA256,
        "split": "last 10% of Unicode characters after floor(0.9 * length)",
        "validation_start_character": len(training),
        "validation_characters": len(validation),
        "scored_text_sha256": hashlib.sha256(text[len(training) + 1:].encode("utf-8")).hexdigest(),
    }
    context = args.context if args.context is not None else model.config.context_length
    quality = score_validation(model, validation, tokenizer, context, args.batch_size, device)
    settings = {
        "count": args.new_tokens, "temperature": args.temperature, "top_k": args.top_k, "seed": args.seed,
    }
    prompts = []
    for prompt in PROMPTS:
        prefix, generated = generate_tokens(model, tokenizer, prompt, device=device, **settings)
        prompts.append({
            "prompt": prompt,
            "text": tokenizer.decode(prefix + generated),
            "continuation": tokenizer.decode(generated),
            "generated_tokens": len(generated),
        })

    config = asdict(model.config)
    result = {
        "schema_version": 1,
        "evaluated_at_utc": datetime.now(UTC).isoformat(),
        "checkpoint": {
            "path": str(path), "sha256": checkpoint_sha, "step": step, "size_bytes": path.stat().st_size,
        },
        "model_config": config,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "tokenizer": {
            "type": "character", "vocab_size": len(tokenizer.vocabulary),
            "sha256": hashlib.sha256(json.dumps(tokenizer.vocabulary, ensure_ascii=False).encode("utf-8")).hexdigest(),
        },
        "dataset": split,
        "scoring": {"protocol": "contiguous_target_blocks_v1", "context_length": context,
                    "batch_size": args.batch_size},
        "generation": {**settings, "prompts": list(PROMPTS)},
        "device": str(device),
        "cpu_threads": torch.get_num_threads(),
        "platform": platform.platform(),
        "torch_version": str(torch.__version__),
        "quality": quality,
        "training": training_metadata(path, step, config, split),
    }
    directory = new_result_dir(args.output_root, f"{path.parent.name}-on-{device.type}")
    write_json(directory / "metrics.json", result)
    write_json(directory / "samples.json", {"generation": result["generation"], "samples": prompts})
    return directory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-root", type=Path, default=EVALUATIONS)
    parser.add_argument("--device", choices=("auto", "mps", "cpu"), default="auto")
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--context", type=int, help="Scoring context, defaults to checkpoint context")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=20)
    args = parser.parse_args()
    if min(args.cpu_threads, args.batch_size, args.new_tokens, args.top_k) < 1:
        parser.error("Thread, batch, token and top-k counts must be positive")
    if not math.isfinite(args.temperature) or args.temperature <= 0:
        parser.error("Temperature must be finite and positive")
    try:
        print(evaluate(args))
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f"Evaluation failed: {error}\n")


if __name__ == "__main__":
    main()
