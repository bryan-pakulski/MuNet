import numpy as np
from munet import nn


class Block(nn.Module):
    def __init__(self, width, heads, rng):
        self.norm1, self.norm2 = nn.LayerNorm(width), nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(width, heads, rng=rng)
        self.mlp = nn.Sequential(nn.Linear(width, 4 * width, rng=rng), nn.GELU(), nn.Linear(4 * width, width, rng=rng))

    def forward(self, x, mask):
        normalized = self.norm1(x)
        attention, _ = self.attention(normalized, normalized, normalized, attn_mask=mask)
        x = x + attention
        return x + self.mlp(self.norm2(x))


class TinyGPT(nn.Module):
    def __init__(self, vocab_size, context=32, width=32, heads=4, layers=2, seed=7):
        if min(vocab_size, context, width, heads, layers) <= 0 or width % heads:
            raise ValueError("use positive model dimensions and width divisible by heads")
        self.context = context
        rng = np.random.default_rng(seed)
        self.tokens = nn.Embedding(vocab_size, width, rng=rng)
        self.positions = nn.Embedding(context, width, rng=rng)
        for embedding in (self.tokens, self.positions):
            embedding.weight.assign(rng.normal(0, .02, embedding.weight.shape).astype(np.float32))
        self.blocks = nn.ModuleList(Block(width, heads, rng) for _ in range(layers))
        self.norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, vocab_size, rng=rng)
        self.mask = np.triu(np.ones((context, context), np.float32), k=1)
        self.position_ids = np.arange(context, dtype=np.float32)[None]

    def forward(self, tokens):
        if tokens.ndim != 2 or tokens.shape[1] != self.context:
            raise ValueError("tokens must have shape (batch, context)")
        x = self.tokens(tokens) + self.positions(self.position_ids)
        for block in self.blocks:
            x = block(x, self.mask)
        return self.head(self.norm(x))
