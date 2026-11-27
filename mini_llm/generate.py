import argparse
import math
from pathlib import Path

import torch

from mini_llm.data import CharTokenizer
from mini_llm.model import GPT, ModelConfig
from mini_llm.runtime import select_device, synchronize


def load_checkpoint(path, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("format_version") != 1 or checkpoint.get("attention") != "causal_sdpa":
        raise ValueError("Unsupported mini_llm checkpoint format or attention backend")
    tokenizer = CharTokenizer(checkpoint["vocabulary"])
    config = ModelConfig(**checkpoint["model_config"])
    if len(tokenizer.vocabulary) != config.vocab_size:
        raise ValueError("Checkpoint vocabulary size does not match the model")
    model = GPT(config)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()
    synchronize(device)
    return model, tokenizer, checkpoint["step"]


@torch.inference_mode()
def generate(model, tokenizer, prompt, count, temperature, top_k, seed, device) -> str:
    tokens = tokenizer.encode(prompt)
    if not tokens:
        raise ValueError("Prompt must contain at least one training-vocabulary character")
    generator = torch.Generator().manual_seed(seed)
    for _ in range(count):
        context = torch.tensor([tokens[-model.config.context_length:]], dtype=torch.long, device=device)
        logits, _ = model(context)
        scores = logits[0, -1].cpu()
        if top_k is not None:
            scores, candidates = scores.topk(min(top_k, scores.numel()))
        # Subtract the maximum before scaling for numerical stability.
        probabilities = ((scores - scores.max()) / temperature).softmax(dim=-1)
        choice = torch.multinomial(probabilities, 1, generator=generator).item()
        tokens.append(choice if top_k is None else candidates[choice].item())
    return tokenizer.decode(tokens)


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample text from a local mini_llm checkpoint.")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--prompt", default="First Citizen:")
    parser.add_argument("--new-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", choices=("auto", "mps", "cpu"), default="auto")
    args = parser.parse_args()
    if args.new_tokens < 0 or not math.isfinite(args.temperature) or args.temperature <= 0:
        parser.error("New token count must be nonnegative and temperature must be finite and positive")
    if args.top_k is not None and args.top_k < 1:
        parser.error("--top-k must be positive")

    try:
        device = select_device(args.device)
        torch.set_num_threads(1)
        model, tokenizer, _ = load_checkpoint(args.checkpoint, device)
        print(generate(model, tokenizer, args.prompt, args.new_tokens, args.temperature, args.top_k, args.seed, device))
    except (OSError, ValueError) as error:
        parser.exit(1, f"Generation failed: {error}\n")


if __name__ == "__main__":
    main()
