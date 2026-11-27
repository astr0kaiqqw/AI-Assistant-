"""Compare evaluation reports and their optional sibling benchmark reports."""

import argparse
import csv
import json
import math
import warnings
from pathlib import Path

from mini_llm.evaluate import EVALUATIONS, new_result_dir

PLOTS = {
    "quality": {
        "validation_loss": "Cross-entropy (nats/token)",
        "perplexity": "Perplexity",
        "bits_per_byte": "Bits per byte",
    },
    "speed": {
        "model_load_seconds": "Model load (s)",
        "time_to_first_token_seconds": "First token (s)",
        "generation_seconds": "Generation (s)",
        "tokens_per_second": "Generated tokens/s",
        "characters_per_second": "Generated characters/s",
    },
    "memory": {
        "process_rss_mib": "Process RSS (MiB)",
        "mps_allocated_mib": "Live MPS tensors (MiB)",
        "checkpoint_size_mib": "Checkpoint file (MiB)",
    },
}


def mib(value):
    return None if value is None else value / (1024 ** 2)


def comparison_rows(paths):
    rows = []
    tokenizer_ids = set()
    scoring_protocols = set()
    generation_protocols = set()
    benchmark_environments = set()
    for index, path in enumerate(paths, 1):
        report = json.loads(path.read_text(encoding="utf-8"))
        checkpoint = report.get("checkpoint") or {}
        tokenizer = report.get("tokenizer") or {}
        dataset = report.get("dataset") or {}
        scoring = report.get("scoring") or {}
        quality = report.get("quality") or {}
        generation = report.get("generation")
        tokenizer_ids.add((tokenizer.get("type"), tokenizer.get("vocab_size"), tokenizer.get("sha256")))
        scoring_protocols.add((dataset.get("sha256"), dataset.get("scored_text_sha256"),
                               scoring.get("protocol"), scoring.get("context_length")))
        generation_protocols.add(json.dumps(generation, sort_keys=True))
        benchmark_path = path.with_name("benchmark.json")
        benchmark = json.loads(benchmark_path.read_text(encoding="utf-8")) if benchmark_path.is_file() else {}
        if benchmark and (benchmark.get("checkpoint_sha256") != checkpoint.get("sha256")
                          or benchmark.get("generation") != generation):
            warnings.warn(f"Ignoring unmatched benchmark at {benchmark_path}")
            benchmark = {}
        if benchmark:
            benchmark_environments.add(tuple(benchmark.get(key) for key in (
                "device", "platform", "torch_version", "cpu_threads", "warmups_per_prompt",
                "repetitions_per_prompt",
            )))
        medians = benchmark.get("medians") or {}
        row = {
            "experiment": f"{index}. {path.resolve().parent.name}",
            "report": str(path.resolve()),
            "checkpoint_sha256": checkpoint.get("sha256"),
            "checkpoint_step": checkpoint.get("step"),
            "tokenizer_type": tokenizer.get("type"),
            "tokenizer_sha256": tokenizer.get("sha256"),
            "vocab_size": tokenizer.get("vocab_size"),
            "parameter_count": report.get("parameter_count"),
            "model_context": (report.get("model_config") or {}).get("context_length"),
            "scoring_context": scoring.get("context_length"),
            "evaluated_tokens": quality.get("evaluated_tokens"),
            "evaluated_bytes": quality.get("evaluated_bytes"),
            "evaluation_device": report.get("device"),
            "benchmark_device": benchmark.get("device"),
            "benchmark_cpu_threads": benchmark.get("cpu_threads"),
        }
        row.update({key: quality.get(key) for key in PLOTS["quality"]})
        row.update({key: medians.get(key) for key in PLOTS["speed"]})
        row["model_load_seconds"] = benchmark.get("model_load_seconds")
        row["process_rss_mib"] = mib(medians.get("process_rss_bytes"))
        row["mps_allocated_mib"] = mib(medians.get("mps_allocated_bytes"))
        row["checkpoint_size_mib"] = mib(benchmark.get("checkpoint_size_bytes", checkpoint.get("size_bytes")))
        rows.append(row)

    if len(tokenizer_ids) > 1 or any(not identifier[2] for identifier in tokenizer_ids):
        warnings.warn("Tokenizer identities differ or are unknown. Perplexity and tokens/s are tokenizer-dependent.")
    if len(scoring_protocols) > 1:
        warnings.warn("Scored text or scoring protocols differ. Quality metrics may be incomparable.")
    if len(generation_protocols) > 1:
        warnings.warn("Generation settings differ. Inference timings and throughput may be incomparable.")
    if len(benchmark_environments) > 1:
        warnings.warn("Benchmark environments or repetition settings differ. Interpret speed and memory in that context.")
    return rows


def plot_reports(rows, directory):
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    for group, fields in PLOTS.items():
        series = {}
        for key, label in fields.items():
            values = [(index, row[key]) for index, row in enumerate(rows, 1)
                      if isinstance(row[key], (float, int)) and math.isfinite(row[key])]
            if values:
                series[label] = values
        if not series:
            continue
        figure, axes = plt.subplots(len(series), 1, figsize=(7, 2.6 * len(series)), squeeze=False)
        for axis, (label, values) in zip(axes[:, 0], series.items()):
            indexes, measurements = zip(*values)
            axis.bar(indexes, measurements)
            axis.set_title(label)
            axis.set_xticks(range(1, len(rows) + 1))
            axis.set_xlabel("Experiment number in comparison.csv")
        figure.tight_layout()
        figure.savefig(directory / f"{group}.png", dpi=140)
        plt.close(figure)


def compare(paths, output_root):
    rows = comparison_rows(paths)
    directory = new_result_dir(output_root, "comparison")
    with (directory / "comparison.csv").open("x", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    plots = directory / "plots"
    plots.mkdir()
    plot_reports(rows, plots)
    return directory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metrics", type=Path, nargs="+", help="Evaluation metrics.json files")
    parser.add_argument("--output-root", type=Path, default=EVALUATIONS)
    args = parser.parse_args()
    if len(args.metrics) < 2:
        parser.error("Provide at least two evaluation reports")
    try:
        print(compare(args.metrics, args.output_root))
    except (OSError, ValueError) as error:
        parser.exit(1, f"Comparison failed: {error}\n")


if __name__ == "__main__":
    main()
