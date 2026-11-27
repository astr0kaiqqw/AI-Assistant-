"""Benchmark the checkpoint and generation settings recorded in an evaluation."""

import argparse
import json
import platform
from datetime import UTC, datetime
from pathlib import Path
from statistics import median
from time import perf_counter

import psutil
import torch

from mini_llm.evaluate import file_sha256, write_json
from mini_llm.generate import generate_tokens, load_checkpoint
from mini_llm.runtime import select_device, synchronize


def memory_snapshot(process, device):
    return {
        "process_rss_bytes": process.memory_info().rss,
        "mps_allocated_bytes": torch.mps.current_allocated_memory() if device.type == "mps" else None,
    }


def generation_trial(model, tokenizer, prompt, settings, device):
    first_token_at = None

    def mark_first_token():
        nonlocal first_token_at
        synchronize(device)
        first_token_at = perf_counter()

    synchronize(device)
    start = perf_counter()
    prefix, generated = generate_tokens(
        model, tokenizer, prompt, device=device, on_first_token=mark_first_token, **settings
    )
    # Decode the continuation inside the total timer.
    characters = len(tokenizer.decode(prefix + generated)) - len(tokenizer.decode(prefix))
    synchronize(device)
    elapsed = perf_counter() - start
    return {
        "prompt": prompt,
        "prompt_tokens": len(prefix),
        "generated_tokens": len(generated),
        "generated_characters": characters,
        "time_to_first_token_seconds": first_token_at - start,
        "generation_seconds": elapsed,
        "tokens_per_second": len(generated) / elapsed,
        "characters_per_second": characters / elapsed,
    }


def benchmark(args):
    report_path = args.metrics.resolve(strict=True)
    output = report_path.with_name("benchmark.json")
    if output.exists():
        raise FileExistsError(output)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    checkpoint = args.checkpoint or Path(report["checkpoint"]["path"])
    if file_sha256(checkpoint) != report["checkpoint"]["sha256"]:
        raise ValueError("Checkpoint SHA-256 differs from the evaluation report")
    device = select_device(args.device or report["device"])
    torch.set_num_threads(args.cpu_threads or report["cpu_threads"])
    settings = dict(report["generation"])
    prompts = settings.pop("prompts")
    torch.manual_seed(settings["seed"])
    process = psutil.Process()
    snapshots = {"before_load": memory_snapshot(process, device)}
    synchronize(device)
    start = perf_counter()
    model, tokenizer, _ = load_checkpoint(checkpoint, device)
    load_seconds = perf_counter() - start
    snapshots["after_load"] = memory_snapshot(process, device)

    for _ in range(args.warmups):
        for prompt in prompts:
            generation_trial(model, tokenizer, prompt, settings, device)
    snapshots["after_warmup"] = memory_snapshot(process, device)

    trials = []
    for repetition in range(1, args.repetitions + 1):
        for prompt in prompts:
            trial = generation_trial(model, tokenizer, prompt, settings, device)
            trials.append({"repetition": repetition, **trial, **memory_snapshot(process, device)})
    medians = {
        key: median(trial[key] for trial in trials)
        for key in ("time_to_first_token_seconds", "generation_seconds", "tokens_per_second",
                    "characters_per_second", "process_rss_bytes", "mps_allocated_bytes")
        if trials[0][key] is not None
    }
    write_json(output, {
        "schema_version": 1,
        "benchmarked_at_utc": datetime.now(UTC).isoformat(),
        "checkpoint_sha256": report["checkpoint"]["sha256"],
        "checkpoint_size_bytes": checkpoint.stat().st_size,
        "device": str(device),
        "cpu_threads": torch.get_num_threads(),
        "torch_version": str(torch.__version__),
        "platform": platform.platform(),
        "generation": report["generation"],
        "warmups_per_prompt": args.warmups,
        "repetitions_per_prompt": args.repetitions,
        "model_load_seconds": load_seconds,
        "memory_snapshots": snapshots,
        "peak_memory_measured": False,
        "medians": medians,
        "trials": trials,
    })
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metrics", type=Path, help="Path to an evaluation's metrics.json")
    parser.add_argument("--checkpoint", type=Path, help="Alternate path to the same checkpoint")
    parser.add_argument("--device", choices=("auto", "mps", "cpu"))
    parser.add_argument("--cpu-threads", type=int)
    parser.add_argument("--warmups", type=int, default=2, help="Warmups per prompt")
    parser.add_argument("--repetitions", type=int, default=5, help="Timed runs per prompt")
    args = parser.parse_args()
    if min(args.warmups, args.repetitions) < 1 or (args.cpu_threads is not None and args.cpu_threads < 1):
        parser.error("Warmups, repetitions and CPU thread counts must be positive")
    try:
        print(benchmark(args))
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f"Benchmark failed: {error}\n")


if __name__ == "__main__":
    main()
