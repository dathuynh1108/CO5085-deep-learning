"""E2: explicit MSA and an identically initialized PyTorch counterpart."""
from __future__ import annotations
import math
import torch
from torch import Tensor, nn
from common.tokenization import image_sequence


class ManualSelfAttention(nn.Module):
    """MSA using Linear, matmul, softmax, dropout, and head concatenation only."""
    def __init__(self, dim: int, heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        if dim <= 0 or heads <= 0 or dim % heads:
            raise ValueError('Embedding dimension must be positive and divisible by heads.')
        self.dim, self.heads, self.head_dim = dim, heads, dim // heads
        self.dropout_p = dropout
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.attention_dropout = nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.qkv.weight)
        nn.init.zeros_(self.qkv.bias)
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: Tensor) -> Tensor:
        b, n, d = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(dim=0)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        weights = self.attention_dropout(scores.softmax(dim=-1))
        context = (weights @ v).transpose(1, 2).reshape(b, n, d)
        return self.proj(context)


class TorchSelfAttention(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.mha = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)

    @classmethod
    def from_manual(cls, source: ManualSelfAttention) -> 'TorchSelfAttention':
        # Do not advance the experiment's RNG just to instantiate the reference module.
        with torch.random.fork_rng(devices=[]):
            target = cls(source.dim, source.heads, source.dropout_p)
        target = target.to(device=source.qkv.weight.device, dtype=source.qkv.weight.dtype)
        with torch.no_grad():
            target.mha.in_proj_weight.copy_(source.qkv.weight)
            target.mha.in_proj_bias.copy_(source.qkv.bias)
            target.mha.out_proj.weight.copy_(source.proj.weight)
            target.mha.out_proj.bias.copy_(source.proj.bias)
        return target

    def forward(self, x: Tensor) -> Tensor:
        return self.mha(x, x, x, need_weights=False)[0]


class ImageTokenizer(nn.Module):
    def __init__(self, kind: str, dim: int, patch_size: int = 4) -> None:
        super().__init__()
        self.kind, self.patch_size = kind, patch_size
        if kind == 'rows':
            self.num_tokens = 28
            self.project = nn.Linear(28, dim)
        elif kind == 'patches':
            if patch_size <= 0 or 28 % patch_size:
                raise ValueError('patch_size must divide 28.')
            self.num_tokens = (28 // patch_size) ** 2
            self.project = nn.Linear(patch_size ** 2, dim)
        elif kind == 'cnn':
            self.num_tokens = 49
            self.project = nn.Sequential(
                nn.Conv2d(1, 32, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(32), nn.ReLU(),
                nn.Conv2d(32, dim, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(dim), nn.ReLU(),
            )
        else:
            raise ValueError(f'Unknown tokenizer: {kind}')

    def forward(self, images: Tensor) -> Tensor:
        if self.kind == 'cnn':
            return self.project(images).flatten(2).transpose(1, 2)
        return self.project(image_sequence(images, self.kind, self.patch_size))


class EncoderBlock(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attention = ManualSelfAttention(dim, heads, dropout)
        self.dropout = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.Linear(dim, 2 * dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(2 * dim, dim), nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.dropout(self.attention(self.norm1(x)))
        return x + self.ffn(self.norm2(x))


class ImageTransformer(nn.Module):
    def __init__(self, tokenizer: str = 'patches', backend: str = 'manual',
                 dim: int = 64, heads: int = 4, depth: int = 2,
                 patch_size: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        if backend not in {'manual', 'torch'} or depth < 1:
            raise ValueError('Invalid attention backend or depth.')
        self.tokenizer = ImageTokenizer(tokenizer, dim, patch_size)
        self.num_tokens = self.tokenizer.num_tokens
        self.position = nn.Parameter(torch.empty(1, self.num_tokens, dim))
        nn.init.normal_(self.position, std=0.02)
        self.input_dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([EncoderBlock(dim, heads, dropout) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, 10)
        # Build the entire manual model first, then copy only the attention modules.
        # Thus every non-attention parameter is also identical for a paired seed.
        if backend == 'torch':
            for block in self.blocks:
                block.attention = TorchSelfAttention.from_manual(block.attention)

    def forward(self, images: Tensor) -> Tensor:
        x = self.input_dropout(self.tokenizer(images) + self.position)
        for block in self.blocks:
            x = block(x)
        return self.head(self.norm(x).mean(dim=1))
