from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    context_length: int = 256
    layers: int = 6
    heads: int = 5
    width: int = 320
    dropout: float = 0.1

    def __post_init__(self):
        if min(self.vocab_size, self.context_length, self.layers, self.heads, self.width) < 1:
            raise ValueError("Model dimensions must be positive")
        if self.width % self.heads:
            raise ValueError("Width must be divisible by the number of heads")
        if not 0 <= self.dropout < 1:
            raise ValueError("Dropout must be in [0, 1)")


class CausalSelfAttention(nn.Module):
    """Map [batch, time, width] to [batch, time, width] with causal attention."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.heads = config.heads
        self.head_width = config.width // config.heads
        self.dropout = config.dropout
        self.qkv = nn.Linear(config.width, 3 * config.width)
        self.output = nn.Linear(config.width, config.width)
        self.residual_dropout = nn.Dropout(config.dropout)

    def forward(self, x: Tensor) -> Tensor:
        batch, time, width = x.shape
        qkv = self.qkv(x).reshape(batch, time, 3, self.heads, self.head_width)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind()
        attended = F.scaled_dot_product_attention(
            q, k, v, is_causal=True,
            dropout_p=self.dropout if self.training else 0.0,
        )
        merged = attended.transpose(1, 2).reshape(batch, time, width)
        return self.residual_dropout(self.output(merged))


class DecoderBlock(nn.Module):
    def __init__(self, config: ModelConfig, attention_factory: Callable[[ModelConfig], nn.Module]):
        super().__init__()
        self.attention_norm = nn.LayerNorm(config.width)
        self.attention = attention_factory(config)
        self.feedforward_norm = nn.LayerNorm(config.width)
        self.feedforward = nn.Sequential(
            nn.Linear(config.width, 4 * config.width),
            nn.GELU(),
            nn.Linear(4 * config.width, config.width),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attention(self.attention_norm(x))
        return x + self.feedforward(self.feedforward_norm(x))


class GPT(nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        attention_factory: Callable[[ModelConfig], nn.Module] = CausalSelfAttention,
    ):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.width)
        self.position_embedding = nn.Embedding(config.context_length, config.width)
        self.embedding_dropout = nn.Dropout(config.dropout)
        self.blocks = nn.Sequential(*(
            DecoderBlock(config, attention_factory) for _ in range(config.layers)
        ))
        self.final_norm = nn.LayerNorm(config.width)
        self.head = nn.Linear(config.width, config.vocab_size, bias=False)
        self.apply(self._initialize)
        self.head.weight = self.token_embedding.weight

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, (nn.Embedding, nn.Linear)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    def forward(self, tokens: Tensor, targets: Tensor | None = None) -> tuple[Tensor, Tensor | None]:
        if tokens.ndim != 2 or not 1 <= tokens.shape[1] <= self.config.context_length:
            raise ValueError("Expected token IDs shaped [batch, time] within the configured context")
        if targets is not None and targets.shape != tokens.shape:
            raise ValueError("Target shape must match token shape")

        positions = torch.arange(tokens.shape[1], device=tokens.device)
        x = self.embedding_dropout(self.token_embedding(tokens) + self.position_embedding(positions))
        logits = self.head(self.final_norm(self.blocks(x)))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, self.config.vocab_size), targets.reshape(-1))
        return logits, loss
