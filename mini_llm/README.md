# Mini LLM Phase 1

An independent PyTorch experiment in character-level language modeling. All
model weights start from random initialization. Training and generation use PyTorch and the Python standard library.
Stage A uses psutil for process memory and Matplotlib for comparison charts.

## Setup

From the repository root, use Python 3.11 or newer with a PyTorch wheel for
your platform. The existing repository environment can work if it has
PyTorch. A dedicated environment is also supported.

```sh
python3 -m venv mini_llm/.venv
source mini_llm/.venv/bin/activate
python -m pip install -r mini_llm/requirements.txt
```

The first training run downloads Tiny Shakespeare from the
[upstream char-rnn dataset](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt)
and checks its SHA-256. The cache is `mini_llm/data/tinyshakespeare.txt`.
Training uses the first 90% of the characters and validation uses the
remaining 10%. The vocabulary is sorted and fitted on the training split.
Input windows and their one-character-shifted targets stay within their split.

## Train

Start the full baseline manually on Apple MPS.

```sh
python -m mini_llm.train --device mps --context 256 --layers 6 --heads 5 --width 320 --batch-size 8 --steps 3000 --warmup 100 --eval-every 500 --eval-batches 16 --seed 1337
```

For a quick smoke run, use:

```sh
python -m mini_llm.train --device auto --batch-size 4 --steps 5 --warmup 1 --eval-every 1 --eval-batches 2 --seed 1337
```

`--device auto` selects MPS when PyTorch reports it available and otherwise
uses CPU. `--device cpu` selects CPU explicitly. An explicit MPS request reports
an error when MPS is unavailable. Training uses float32, AdamW, linear warmup
followed by cosine decay, and gradient-norm clipping at 1.0. The default CPU
thread count is one. Adjust it with `--cpu-threads` when benchmarking CPU.

Each run gets its own ignored directory under `mini_llm/runs/`. If you pass
`--run-dir`, choose a new directory. The run saves:

- `config.json` with model and training settings, dataset checksum, split
  sizes, device, PyTorch version, and parameter count.
- `tokenizer.json` with the character vocabulary in token-ID order.
- `metrics.csv` with train loss, sampled validation loss, learning rate, and
  synchronized training-step time. Validation runs at evaluation steps and at
  the end. Each evaluation uses the same seeded validation-window sample.
- `checkpoint.pt` with model and optimizer state, vocabulary, architecture,
  and step number. The checkpoint is updated at evaluation steps. The
  generation CLI loads it for inference. Training resume is a future feature.

Step timing includes sampling, transfer, forward and backward passes,
clipping, and the optimizer update. Evaluation, CSV writing, and checkpoint
saving happen outside that interval. MPS synchronizes around each measured
step. Seeds control initialization, sampling, and dropout where supported.
Exact numbers may differ across devices and PyTorch versions.

## Generate

Use the checkpoint path printed by the training command.

```sh
python -m mini_llm.generate mini_llm/runs/RUN_ID/checkpoint.pt \
  --device auto --prompt 'First Citizen:' --new-tokens 200 --temperature 0.8 --top-k 20
```

The prompt must use characters in the saved vocabulary. Generation samples
one character at a time and uses at most the last 256 characters with the
default configuration. Sampling is seeded. The baseline recomputes the
context each time and has no key-value cache.

## Stage A evaluation and benchmarking

Use an existing checkpoint and the local checksum-verified dataset cache.
These commands create fresh result directories under
`mini_llm/evaluations/`. Run each benchmark as its own CLI process.

```sh
first=$(python -m mini_llm.evaluate mini_llm/runs/RUN_A/checkpoint.pt --device auto)
python -m mini_llm.benchmark "$first/metrics.json"

second=$(python -m mini_llm.evaluate mini_llm/runs/RUN_B/checkpoint.pt --device auto)
python -m mini_llm.benchmark "$second/metrics.json"

python -m mini_llm.compare "$first/metrics.json" "$second/metrics.json"
python -m unittest -v mini_llm.test_evaluation
```

An evaluation directory contains `metrics.json`, `samples.json`, and
`benchmark.json` after benchmarking. The comparison command writes
`comparison.csv` and the available quality, speed, and memory charts in
`plots/` under a separate timestamped directory. `--output-root` changes
the evaluation or comparison destination. To rebenchmark with different
settings, create a new evaluation directory. An existing benchmark is
preserved. Use `--checkpoint` on the benchmark command if the same checkpoint
was moved. Its SHA-256 must match the report.

### Validation protocol

The evaluator reconstructs the model and ordered character vocabulary
from the checkpoint and checks that vocabulary against the one fitted to
the original training split. It scores the held-out tail in order using
the original causal next-token objective. Windows have nonoverlapping
targets. Each window starts with the character immediately before its
first target and resets positional indices. The final partial window
is included. The first validation character supplies context only,
giving 111,539 scored targets for Tiny Shakespeare.

- **Validation loss** is the sum of target negative log-likelihoods in
  nats divided by the number of scored tokens. The shorter final window
  contributes in proportion to its target count.
- **Perplexity** is the exponential of the validation loss and depends
  on the token units used by the model.
- **Bits per byte** divides the same total negative log-likelihood by
  ln(2) and the UTF-8 byte count of the exact scored text. The character
  tokenizer has a lossless mapping to this text. The first validation
  character is excluded from both totals.

This is a blockwise full-split score. Context shrinks at block boundaries.
The sampled validation loss in training logs uses a different protocol.
The logged training time is the sum of recorded training-step durations
through the saved step and excludes validation and checkpoint I/O.
Reports record model and scoring context lengths separately, along with the
checkpoint, tokenizer and scored-text hashes. Use identical scoring context
(`--context`, no greater than the smallest checkpoint context) for matched
character-model comparisons. Keep the fixed prompts, seed, generation
settings, device and thread count consistent for controlled measurements.
Defaults are three fixed prompts, 64 new tokens per prompt, temperature
0.8, top-k 20, and seed 1337 reset for each prompt. Sampling is reproducible
within a matching execution environment. Device and library changes can
change numerical results or samples.

### Benchmark protocol

The benchmark reads its prompts and sampling settings from `metrics.json`.
It runs two warmups and five measured repetitions per prompt by default.
The report excludes warmups and stores each measured trial and the median
across all measured prompt-trials.
Load time is measured once per invocation and includes checkpoint
deserialization, model construction, weight loading and device transfer.
The checkpoint is hashed before timing, so load timing can benefit from
the filesystem cache. The checkpoint size includes optimizer state.

Time to first token starts before prompt encoding and ends when the first
sampled token ID is ready. Total generation time also includes decoding.
Throughput divides generated tokens or generated Unicode characters by
total generation time and excludes the prompt from the numerator. MPS
synchronizes at timing boundaries. The generation path samples on CPU and
recomputes context on every token.

Process RSS from psutil is resident process memory. MPS allocated memory
is the amount held by live PyTorch MPS tensors. Neither is a measurement
of peak memory or total unified memory usage. Snapshots are taken before
loading, after loading, after warmup and after each timed trial. The
comparison shows median post-trial snapshots. CPU reports omit MPS
measurements and the comparison leaves unavailable values blank.

### Measured Stage A reference

The MPS-trained checkpoint was saved after five smoke-training steps.
The model has 7,501,120 trainable parameters, six pre-normalized decoder
blocks, five causal SDPA heads, width 320, 256 positions, and tied input
and output embeddings. Its character vocabulary has 65 entries. Full-split
evaluation on MPS scored 111,539 tokens across 111,539 UTF-8 bytes, with
cross-entropy **3.441037 nats/token**, perplexity **31.2193**, and
**4.964366 bits/byte**.

Both benchmark columns use that same checkpoint on an Apple M4 with
16 GB unified memory and PyTorch 2.15.0. Settings were seed 1337,
temperature 0.8, top-k 20, 64 new tokens per prompt, two warmups and five
measured repetitions per prompt. The three prompts were `First Citizen:`,
`ROMEO:`, and `To be, or not to be`.

| Metric | CPU (one thread) | Apple MPS |
| --- | ---: | ---: |
| Model load (single measurement) | 105.8 ms | 121.5 ms |
| First token (median) | 2.38 ms | 2.22 ms |
| Total generation (median) | 192.95 ms | 145.44 ms |
| Generated tokens/s (median) | 331.7 | 440.1 |
| Generated characters/s (median) | 331.7 | 440.1 |
| Process RSS (median post-trial) | 342.6 MiB | 466.3 MiB |
| Live MPS tensor allocation (median post-trial) | — | 28.7 MiB |
| Checkpoint file | 85.9 MiB | 85.9 MiB |

MPS delivered approximately **1.33×** the CPU throughput on this workload.
Peak memory was not measured. Process RSS and live MPS allocation
describe separate quantities. The model-load measurement may benefit
from filesystem caching.

All six Stage A tests passed, including CPU and MPS smoke evaluation.
A repeated CPU evaluation reproduced the quality metrics and sample JSON
exactly. Checkpoint hashes remained unchanged. Both available checkpoints
were trained for five smoke steps. These results establish a pipeline and
inference performance reference. The full-training quality baseline is pending.

### Before Stage B

Keep the pinned raw corpus and its character split boundary. Fit BPE on the
training text only, persist its complete tokenizer artifact and fingerprint,
and extend the checkpoint loader to reconstruct it. Define and verify a
shared scored UTF-8 span before comparing bits per byte across tokenizers.
Account explicitly for first-token context, special tokens and byte
boundaries. Report context in tokens and document how much raw text each
context covers. For inference, keep prompts and sampling settings fixed
and record any output-budget difference caused by tokenizer granularity.
Use characters per second alongside tokens per second. Plan matched
training budgets and a fully trained character reference before drawing
quality conclusions. The comparison tool accepts tokenizer identities
and warns when perplexity or token throughput spans different tokenizers.

## Architecture and comparison

The default baseline has 7,501,120 trainable parameters. It uses a
65-character vocabulary, 256 learned positions,
six pre-normalized decoder blocks, a width of 320, and five attention heads
of width 64. Each block adds attention and a 4× GELU feed-forward network
through separate residual connections. A final layer norm feeds a
next-character prediction head that shares the token embedding weights.

The fused projection produces query, key, and value vectors for each token
and head. Each query scores keys using a scaled dot product. The causal mask
excludes positions to the right of the query before softmax. Each output is a
weighted sum of the allowed values. Training cross-entropy compares the
logits at each position with the following character.

`GPT(config, attention_factory=...)` accepts a factory producing a module that
maps `[batch, time, width]` back to that shape with causal behavior. A later
comparison can swap the attention module while keeping the rest of the
model, tokenizer, data windows, and timing procedure fixed. Record parameter
counts, device, batch size, context length, precision, and data budget for
each variant. Standard attention has quadratic compute in context length.
The five-step smoke run verifies mechanics and timing, with too few steps
for conclusions about text quality.

## Verified smoke results

Two five-step runs completed on an Apple M4 Mac with 16 GB of unified
memory and PyTorch 2.15.0. Both used float32, batch size 4, context 256,
and seed 1337. The checksum-verified corpus yielded 1,003,854 training and
111,540 validation characters. Each evaluation used two fixed sampled
validation batches.

| Device | Train loss, step 1 → 5 | Validation loss, step 1 → 5 | First step | Median of steps 2–5 |
| --- | ---: | ---: | ---: | ---: |
| MPS | 4.2527 → 3.3708 | 3.6892 → 3.5028 | 2.1087 s | 43.4 ms |
| CPU, one thread | 4.2337 → 3.3693 | 3.6844 → 3.5043 | 0.1580 s | 137.2 ms |

Both runs produced `[4, 256, 65]` logits, finite losses and nonzero
gradients. Both step-five checkpoints loaded and generated text with
temperature and top-k sampling. Training backpropagates cross-entropy
through the model before clipping gradients and applying the AdamW update.

## Limitations and next steps

Only smoke training is complete. The samples demonstrate checkpoint
inference. Text quality and conversational capability have yet to be
established, and the model is unsuitable for production chat.

1. Run full baseline pretraining and analyze validation loss over time.
2. Implement a linear-attention variant and compare it with standard causal
   attention under matched training and measurement conditions.
3. Evaluate supervised fine-tuning (SFT) and possible model routing and
   integration with AI Assistant after the baseline comparison.
