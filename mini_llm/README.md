# Mini LLM Phase 1

An independent PyTorch experiment in character-level language modeling. All
model weights start from random initialization. The experiment uses only
PyTorch and the Python standard library.

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
