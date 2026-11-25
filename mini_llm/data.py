from hashlib import sha256
from pathlib import Path
from urllib.request import urlopen

import torch
from torch import Tensor


DATASET_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DATASET_SHA256 = "86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed"
DATASET_BYTES = 1_115_394
PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA = PACKAGE_DIR / "data" / "tinyshakespeare.txt"


class CharTokenizer:
    def __init__(self, vocabulary: list[str]):
        if not vocabulary or len(set(vocabulary)) != len(vocabulary) or any(len(c) != 1 for c in vocabulary):
            raise ValueError("Expected a nonempty vocabulary of unique single characters")
        self.vocabulary = vocabulary
        self.ids = {char: index for index, char in enumerate(vocabulary)}

    def encode(self, text: str) -> list[int]:
        try:
            return [self.ids[char] for char in text]
        except KeyError as error:
            raise ValueError(f"Character {error.args[0]!r} is outside the training vocabulary") from None

    def decode(self, tokens: list[int]) -> str:
        return "".join(self.vocabulary[token] for token in tokens)


def load_dataset(path: Path) -> str:
    if path.exists():
        if path.stat().st_size != DATASET_BYTES:
            raise ValueError(f"Unexpected dataset size at {path}. Choose a fresh --data path.")
        raw = path.read_bytes()
    else:
        with urlopen(DATASET_URL, timeout=30) as response:
            raw = response.read(DATASET_BYTES + 1)
        _verify_dataset(raw)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as file:
            file.write(raw)
        return raw.decode("utf-8")

    _verify_dataset(raw)
    return raw.decode("utf-8")


def _verify_dataset(raw: bytes) -> None:
    if sha256(raw).hexdigest() != DATASET_SHA256:
        raise ValueError("Tiny Shakespeare checksum mismatch")


def dataset_splits(text: str) -> tuple[CharTokenizer, Tensor, Tensor]:
    boundary = len(text) * 9 // 10
    training, validation = text[:boundary], text[boundary:]
    tokenizer = CharTokenizer(sorted(set(training)))
    return (
        tokenizer,
        torch.tensor(tokenizer.encode(training), dtype=torch.long),
        torch.tensor(tokenizer.encode(validation), dtype=torch.long),
    )


def sample_batch(
    data: Tensor, batch_size: int, context: int, generator: torch.Generator, device: torch.device
) -> tuple[Tensor, Tensor]:
    starts = torch.randint(len(data) - context, (batch_size,), generator=generator, device="cpu")
    indices = starts[:, None] + torch.arange(context + 1)
    windows = data[indices].to(device)
    return windows[:, :-1], windows[:, 1:]
