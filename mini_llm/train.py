import argparse
import csv
import json
import math
import platform
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from statistics import median
from time import perf_counter

import torch

from mini_llm.data import (
    DATASET_SHA256, DATASET_URL, DEFAULT_DATA, PACKAGE_DIR, dataset_splits, load_dataset,
    sample_batch,
)
from mini_llm.model import GPT, ModelConfig
from mini_llm.runtime import select_device, synchronize


def learning_rate(step: int, steps: int, warmup: int, peak: float, floor: float) -> float:
    if step <= warmup:
        return peak * step / warmup
    progress = (step - max(warmup, 1)) / max(steps - max(warmup, 1), 1)
    return floor + (peak - floor) * (1 + math.cos(math.pi * progress)) / 2


@torch.no_grad()
def validation_loss(model, data, args, device) -> float:
    model.eval()
    generator = torch.Generator().manual_seed(args.seed + 1)
    losses = []
    for _ in range(args.eval_batches):
        inputs, targets = sample_batch(data, args.batch_size, args.context, generator, device)
        _, loss = model(inputs, targets)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def save_checkpoint(run_dir, model, optimizer, vocabulary, step) -> None:
    temporary = run_dir / "checkpoint.tmp"
    torch.save({
        "format_version": 1,
        "attention": "causal_sdpa",
        "model_config": asdict(model.config),
        "vocabulary": vocabulary,
        "step": step,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
    }, temporary)
    temporary.replace(run_dir / "checkpoint.pt")


def train(args) -> None:
    device = select_device(args.device)
    torch.set_num_threads(args.cpu_threads)
    torch.manual_seed(args.seed)
    sampler = torch.Generator().manual_seed(args.seed)
    tokenizer, training, validation = dataset_splits(load_dataset(args.data))
    config = ModelConfig(
        len(tokenizer.vocabulary), args.context, args.layers, args.heads, args.width, args.dropout
    )
    if args.context >= min(len(training), len(validation)):
        raise ValueError("Context must be shorter than both dataset splits")
    model = GPT(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1, foreach=False)

    run_dir = args.run_dir
    if run_dir is None:
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
        run_dir = PACKAGE_DIR / "runs" / stamp
    run_dir.mkdir(parents=True, exist_ok=False)
    params = sum(parameter.numel() for parameter in model.parameters())
    metadata = {
        "model": asdict(config),
        "training": {**vars(args), "data": str(args.data), "run_dir": str(run_dir)},
        "dataset": {
            "source": DATASET_URL, "sha256": DATASET_SHA256,
            "train_characters": len(training), "validation_characters": len(validation),
        },
        "device": str(device),
        "torch": str(torch.__version__),
        "platform": platform.platform(),
        "attention": "causal_sdpa",
        "parameters": params,
    }
    (run_dir / "config.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    (run_dir / "tokenizer.json").write_text(
        json.dumps({"type": "character", "vocabulary": tokenizer.vocabulary}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Device: {device} | Parameters: {params:,} | Run: {run_dir}", flush=True)
    print(f"Dataset: {len(training):,} train / {len(validation):,} validation characters", flush=True)

    times = []
    model.train()
    with (run_dir / "metrics.csv").open("x", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(("step", "train_loss", "validation_loss", "learning_rate", "step_seconds"))
        for step in range(1, args.steps + 1):
            rate = learning_rate(step, args.steps, args.warmup, args.lr, args.min_lr)
            optimizer.param_groups[0]["lr"] = rate
            synchronize(device)
            start = perf_counter()
            inputs, targets = sample_batch(training, args.batch_size, args.context, sampler, device)
            optimizer.zero_grad(set_to_none=True)
            logits, loss = model(inputs, targets)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            synchronize(device)
            elapsed = perf_counter() - start
            times.append(elapsed)
            train_loss = loss.item()
            if step == 1:
                print(
                    f"Shapes: X {tuple(inputs.shape)}, Y {tuple(targets.shape)}, "
                    f"logits {tuple(logits.shape)} | initial gradient norm {norm.item():.4f}",
                    flush=True,
                )

            val = None
            if step % args.eval_every == 0 or step == args.steps:
                val = validation_loss(model, validation, args, device)
                print(f"Step {step}: train={train_loss:.4f} val={val:.4f} lr={rate:.3g}", flush=True)
            writer.writerow((step, train_loss, "" if val is None else val, rate, elapsed))
            file.flush()
            if val is not None:
                save_checkpoint(run_dir, model, optimizer, tokenizer.vocabulary, step)

    typical = median(times[1:] or times)
    tokens_per_second = args.batch_size * args.context / typical
    print(
        f"Training steps: first={times[0]:.4f}s median after first={typical:.4f}s "
        f"({tokens_per_second:,.0f} tokens/s)",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a randomly initialized character-level GPT.")
    parser.add_argument("--device", choices=("auto", "mps", "cpu"), default="auto")
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--run-dir", type=Path, help="New directory for run artifacts")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--context", type=int, default=256)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--heads", type=int, default=5)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min-lr", type=float, default=3e-5)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--eval-batches", type=int, default=16)
    args = parser.parse_args()
    if min(args.steps, args.batch_size, args.eval_every, args.eval_batches, args.cpu_threads) < 1:
        parser.error("Steps, batch size, evaluation settings and CPU threads must be positive")
    if not 0 <= args.warmup < args.steps:
        parser.error("--warmup must be between zero and steps - 1")
    if not math.isfinite(args.lr) or not 0 <= args.min_lr <= args.lr or args.lr <= 0:
        parser.error("Learning rates must be finite with 0 <= min-lr <= lr and lr > 0")
    try:
        train(args)
    except (OSError, ValueError) as error:
        parser.exit(1, f"Training could not start or complete: {error}\n")


if __name__ == "__main__":
    main()
