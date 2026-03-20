import argparse
import json
import os
import shutil
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "../../build"))
import munet

DEFAULT_SYSTEM_PROMPT = "You are a helpful tiny MuNet assistant."
DEFAULT_CORPUS = """
System: You are a helpful tiny MuNet assistant.
User: hello
Assistant: Hello! I am a tiny GPT-style demo built with MuNet.
System: You are a helpful tiny MuNet assistant.
User: what can you do
Assistant: I can answer simple questions, explain MuNet demos, and generate short replies.
System: You are a helpful tiny MuNet assistant.
User: explain transformers
Assistant: Transformers mix token embeddings, attention, normalization, and feed-forward layers to predict the next token.
System: You are a helpful tiny MuNet assistant.
User: how do I train locally
Assistant: Start with a small corpus, train for a few hundred steps, save the weights, then load them for generation or chat.
System: You are a helpful tiny MuNet assistant.
User: what is model parallelism
Assistant: Model parallelism places different layers on different devices and moves activations between them during forward and backward passes.
System: You are a helpful tiny MuNet assistant.
User: tell me a joke
Assistant: Why did the tiny language model stay calm? Because it had good normalization.
System: You are a helpful tiny MuNet assistant.
User: summarize munet
Assistant: MuNet is a lightweight framework with tensors, autograd, multiple backends, and small training and inference demos.
""".strip()
SAMPLING_PRESETS: Dict[str, Dict[str, float]] = {
    "deterministic": {
        "temperature": 0.2,
        "top_k": 1,
        "top_p": 1.0,
        "repetition_penalty": 1.02,
    },
    "balanced": {
        "temperature": 0.8,
        "top_k": 8,
        "top_p": 0.92,
        "repetition_penalty": 1.08,
    },
    "creative": {
        "temperature": 1.0,
        "top_k": 16,
        "top_p": 0.96,
        "repetition_penalty": 1.12,
    },
}


def set_seed(seed: int):
    np.random.seed(seed)


def same_device(lhs, rhs):
    return lhs.type == rhs.type and lhs.index == rhs.index


def discover_accelerator_devices(max_indices: int = 8) -> List["munet.Device"]:
    devices = []
    for device_type in (munet.DeviceType.CUDA, munet.DeviceType.VULKAN):
        for index in range(max_indices):
            dev = munet.Device(device_type, index)
            try:
                munet.ones([1], device=dev)
                devices.append(dev)
            except RuntimeError:
                break
    return devices


def resolve_shard_devices(requested_shards: int) -> List["munet.Device"]:
    accelerators = discover_accelerator_devices()
    if len(accelerators) >= requested_shards:
        return accelerators[:requested_shards]
    if accelerators:
        expanded = [
            accelerators[i % len(accelerators)] for i in range(requested_shards)
        ]
        print(
            f"Only found {len(accelerators)} accelerator(s); reusing them for {requested_shards} shard(s): "
            + ", ".join(repr(dev) for dev in expanded)
        )
        return expanded
    cpu = munet.Device(munet.DeviceType.CPU, 0)
    fallback = [cpu for _ in range(requested_shards)]
    print(
        "No accelerators detected; running the sharded GPT demo on CPU fallback shards."
    )
    return fallback


class CharacterTokenizer:
    def __init__(self, chars: List[str]):
        self.chars = list(chars)
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for ch, i in self.stoi.items()}
        self.pad_token = " " if " " in self.stoi else self.chars[0]

    @classmethod
    def from_text(cls, text: str):
        return cls(sorted(set(text)))

    @property
    def vocab_size(self) -> int:
        return len(self.chars)

    def sanitize(self, text: str) -> str:
        return "".join(ch if ch in self.stoi else self.pad_token for ch in text)

    def encode(self, text: str) -> np.ndarray:
        return np.array([self.stoi[ch] for ch in self.sanitize(text)], dtype=np.int32)

    def decode(self, ids: np.ndarray) -> str:
        return "".join(self.itos[int(idx)] for idx in ids)


class FeedForwardGELU(munet.nn.Module):
    def __init__(self, d_model: int, hidden: int, options):
        super().__init__()
        self.fc1 = munet.nn.Linear(d_model, hidden, True, options=options)
        self.act = munet.nn.GELU()
        self.fc2 = munet.nn.Linear(hidden, d_model, True, options=options)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class FeedForwardSwiGLU(munet.nn.Module):
    def __init__(self, d_model: int, hidden: int, options):
        super().__init__()
        self.gate = munet.nn.Linear(d_model, hidden, True, options=options)
        self.value = munet.nn.Linear(d_model, hidden, True, options=options)
        self.proj = munet.nn.Linear(hidden, d_model, True, options=options)

    def forward(self, x):
        gate = self.gate(x)
        silu = gate * gate.sigmoid()
        return self.proj(silu * self.value(x))


def build_norm(norm_type: str, d_model: int, opts):
    if norm_type == "rmsnorm":
        return munet.nn.RMSNorm(d_model, options=opts)
    return munet.nn.LayerNorm(d_model, options=opts)


class RotarySelfAttention(munet.nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        ctx: int,
        device,
        use_rope: bool,
        attn_dropout: float,
    ):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        opts = munet.TensorOptions()
        opts.device = device
        self.device = device
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.max_ctx = ctx
        self.use_rope = use_rope
        self.q_proj = munet.nn.Linear(d_model, d_model, True, options=opts)
        self.k_proj = munet.nn.Linear(d_model, d_model, True, options=opts)
        self.v_proj = munet.nn.Linear(d_model, d_model, True, options=opts)
        self.out_proj = munet.nn.Linear(d_model, d_model, True, options=opts)
        self.attn_drop = munet.nn.Dropout(attn_dropout)
        self._rope_cache = {}
        self.reset_cache()

    def reset_cache(self):
        self.cache_k = None
        self.cache_v = None

    def cache_len(self) -> int:
        return 0 if self.cache_k is None else int(self.cache_k.shape[2])

    def _rope_tables(self, seq_len: int, device):
        key = (seq_len, device.type, device.index)
        if key in self._rope_cache:
            return self._rope_cache[key]
        half = self.head_dim // 2
        base = 10000.0
        inv_freq = 1.0 / (base ** (np.arange(half, dtype=np.float32) / max(1, half)))
        positions = np.arange(seq_len, dtype=np.float32)[:, None]
        angles = positions * inv_freq[None, :]
        cos = munet.from_numpy(
            np.cos(angles).reshape(1, 1, seq_len, half).astype(np.float32)
        ).to(device)
        sin = munet.from_numpy(
            np.sin(angles).reshape(1, 1, seq_len, half).astype(np.float32)
        ).to(device)
        self._rope_cache[key] = (cos, sin)
        return cos, sin

    def _rope_components(self, seq_len: int):
        half = self.head_dim // 2
        base = 10000.0
        inv_freq = 1.0 / (base ** (np.arange(half, dtype=np.float32) / max(1, half)))
        positions = np.arange(seq_len, dtype=np.float32)[:, None]
        angles = positions * inv_freq[None, :]
        return np.cos(angles).astype(np.float32), np.sin(angles).astype(np.float32)

    def _apply_rope(self, x):
        if not self.use_rope:
            return x
        if self.head_dim % 2 != 0:
            raise ValueError("RoPE requires an even head dimension")
        cos, sin = self._rope_tables(x.shape[2], x.device)
        half = self.head_dim // 2
        x1 = x.narrow(3, 0, half).contiguous()
        x2 = x.narrow(3, half, half).contiguous()
        return munet.Tensor.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], 3)

    def _apply_rope_numpy(self, x_np: np.ndarray, position_index: int) -> np.ndarray:
        if not self.use_rope:
            return x_np
        half = self.head_dim // 2
        cos, sin = self._rope_components(position_index + 1)
        cos_row = cos[position_index].reshape(1, 1, 1, half)
        sin_row = sin[position_index].reshape(1, 1, 1, half)
        x1 = x_np[..., :half]
        x2 = x_np[..., half:]
        return np.concatenate(
            [x1 * cos_row - x2 * sin_row, x1 * sin_row + x2 * cos_row], axis=-1
        )

    def _project_qkv(self, x):
        bsz, seq_len, _ = x.shape
        flat = x.reshape([bsz * seq_len, self.d_model])
        q = (
            self.q_proj(flat)
            .reshape([bsz, seq_len, self.n_heads, self.head_dim])
            .permute([0, 2, 1, 3])
            .contiguous()
        )
        k = (
            self.k_proj(flat)
            .reshape([bsz, seq_len, self.n_heads, self.head_dim])
            .permute([0, 2, 1, 3])
            .contiguous()
        )
        v = (
            self.v_proj(flat)
            .reshape([bsz, seq_len, self.n_heads, self.head_dim])
            .permute([0, 2, 1, 3])
            .contiguous()
        )
        return q, k, v

    def forward(self, x):
        bsz, seq_len, _ = x.shape
        q, k, v = self._project_qkv(x)
        q = self._apply_rope(q)
        k = self._apply_rope(k)
        q2 = q.reshape([bsz * self.n_heads * seq_len, self.head_dim])
        k2 = k.reshape([bsz * self.n_heads * seq_len, self.head_dim])
        v2 = v.reshape([bsz * self.n_heads * seq_len, self.head_dim])
        scores = q2 @ k2.transpose(0, 1)
        scores = scores * (1.0 / np.sqrt(float(self.head_dim)))
        cpu = munet.Device(munet.DeviceType.CPU, 0)
        mask_cpu = munet.Tensor(
            [bsz * self.n_heads * seq_len, bsz * self.n_heads * seq_len],
            device=cpu,
            dtype=munet.DataType.Int32,
        )
        mask_arr = np.array(mask_cpu, copy=False)
        for i in range(bsz * self.n_heads * seq_len):
            bh_i = i // seq_len
            t_i = i % seq_len
            for j in range(bsz * self.n_heads * seq_len):
                bh_j = j // seq_len
                t_j = j % seq_len
                mask_arr[i, j] = 1 if (bh_i != bh_j or t_j > t_i) else 0
        scores = scores.masked_fill(mask_cpu.to(scores.device), -1e9)
        probs = self.attn_drop(scores.softmax(-1))
        ctx = probs @ v2
        merged = (
            ctx.reshape([bsz, self.n_heads, seq_len, self.head_dim])
            .permute([0, 2, 1, 3])
            .contiguous()
            .reshape([bsz * seq_len, self.d_model])
        )
        return self.out_proj(merged).reshape([bsz, seq_len, self.d_model])

    def forward_cached(self, x, position_index: int):
        bsz, seq_len, _ = x.shape
        if bsz != 1 or seq_len != 1:
            raise ValueError("KV-cache path expects [1, 1, E] inputs")
        q, k, v = self._project_qkv(x)
        q_np = tensor_to_numpy_copy(q).reshape(1, self.n_heads, 1, self.head_dim)
        k_np = tensor_to_numpy_copy(k).reshape(1, self.n_heads, 1, self.head_dim)
        v_np = tensor_to_numpy_copy(v).reshape(1, self.n_heads, 1, self.head_dim)
        q_np = self._apply_rope_numpy(q_np, position_index)
        k_np = self._apply_rope_numpy(k_np, position_index)
        if self.cache_k is None:
            self.cache_k = k_np
            self.cache_v = v_np
        else:
            self.cache_k = np.concatenate([self.cache_k, k_np], axis=2)
            self.cache_v = np.concatenate([self.cache_v, v_np], axis=2)
            if self.cache_k.shape[2] > self.max_ctx:
                self.cache_k = self.cache_k[:, :, -self.max_ctx :, :]
                self.cache_v = self.cache_v[:, :, -self.max_ctx :, :]
        scale = 1.0 / np.sqrt(float(self.head_dim))
        scores = np.einsum("bhtd,bhsd->bhts", q_np, self.cache_k) * scale
        scores = scores - np.max(scores, axis=-1, keepdims=True)
        probs = np.exp(scores)
        probs /= np.maximum(np.sum(probs, axis=-1, keepdims=True), 1e-12)
        ctx_np = np.einsum("bhts,bhsd->bhtd", probs, self.cache_v).astype(np.float32)
        ctx = munet.from_numpy(ctx_np.reshape(1, self.n_heads, 1, self.head_dim)).to(
            x.device
        )
        merged = ctx.permute([0, 2, 1, 3]).contiguous().reshape([1, self.d_model])
        return self.out_proj(merged).reshape([1, 1, self.d_model])


class GPTDecoderBlock(munet.nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        ff_hidden: int,
        ffn_type: str,
        norm_type: str,
        position_type: str,
        ctx: int,
        device,
        attn_dropout: float,
        resid_dropout: float,
    ):
        super().__init__()
        opts = munet.TensorOptions()
        opts.device = device
        self.shard_device = device
        self.ln1 = build_norm(norm_type, d_model, opts)
        self.attn = RotarySelfAttention(
            d_model, n_heads, ctx, device, position_type == "rope", attn_dropout
        )
        self.attn_drop = munet.nn.Dropout(attn_dropout)
        self.ln2 = build_norm(norm_type, d_model, opts)
        self.resid_drop = munet.nn.Dropout(resid_dropout)
        self.ff = (
            FeedForwardSwiGLU(d_model, ff_hidden, opts)
            if ffn_type == "swiglu"
            else FeedForwardGELU(d_model, ff_hidden, opts)
        )

    def forward(self, x):
        if not same_device(x.device, self.shard_device):
            x = x.to(self.shard_device)
        h = self.ln1(x)
        x = x + self.attn_drop(self.attn(h))
        h = self.ln2(x)
        flat = h.reshape([h.shape[0] * h.shape[1], h.shape[2]])
        ff = self.ff(flat).reshape([h.shape[0], h.shape[1], h.shape[2]])
        return x + self.resid_drop(ff)

    def forward_cached(self, x, position_index: int):
        if not same_device(x.device, self.shard_device):
            x = x.to(self.shard_device)
        h = self.ln1(x)
        x = x + self.attn.forward_cached(h, position_index)
        h = self.ln2(x)
        ff = self.ff(h.reshape([1, h.shape[2]])).reshape([1, 1, h.shape[2]])
        return x + ff


class FullGPTDemoModel(munet.nn.Module):
    """A trainable GPT-3-style decoder LM with optional model sharding.
    Current architecture can now toggle between LayerNorm and RMSNorm, and
    between learned positional embeddings and rotary position embeddings,
    while keeping the same tiny sharded GPT-style workflow.
    """

    def __init__(
        self,
        vocab: int,
        ctx: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        ffn_type: str,
        shard_devices: List["munet.Device"],
        attn_dropout: float = 0.0,
        resid_dropout: float = 0.0,
        embed_dropout: float = 0.0,
        norm_type: str = "layernorm",
        position_type: str = "learned",
    ):
        super().__init__()
        self.vocab = vocab
        self.ctx = ctx
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.ffn_type = ffn_type
        self.shard_devices = shard_devices
        self.attn_dropout = attn_dropout
        self.resid_dropout = resid_dropout
        self.embed_dropout = embed_dropout
        self.norm_type = norm_type
        self.position_type = position_type
        first_opts = munet.TensorOptions()
        first_opts.device = shard_devices[0]
        last_opts = munet.TensorOptions()
        last_opts.device = shard_devices[-1]
        self.token = munet.nn.Embedding(vocab, d_model, options=first_opts)
        self.pos = (
            munet.nn.Embedding(ctx, d_model, options=first_opts)
            if position_type == "learned"
            else None
        )
        self.embed_drop = munet.nn.Dropout(embed_dropout)
        self.blocks = []
        ff_hidden = 4 * d_model
        for layer in range(n_layers):
            device = shard_devices[layer % len(shard_devices)]
            block = GPTDecoderBlock(
                d_model=d_model,
                n_heads=n_heads,
                ff_hidden=ff_hidden,
                ffn_type=ffn_type,
                norm_type=norm_type,
                position_type=position_type,
                ctx=ctx,
                device=device,
                attn_dropout=attn_dropout,
                resid_dropout=resid_dropout,
            )
            setattr(self, f"block_{layer}", block)
            self.blocks.append(block)
        self.ln_f = build_norm(norm_type, d_model, last_opts)
        self.head = munet.nn.Linear(d_model, vocab, True, options=last_opts)

    @property
    def output_device(self):
        return self.shard_devices[-1]

    def reset_kv_cache(self):
        self.cached_tokens = 0
        for block in self.blocks:
            block.attn.reset_cache()

    def forward(self, tok_oh, pos_oh):
        first = self.shard_devices[0]
        x = self.token(tok_oh.to(first))
        if self.pos is not None:
            x = x + self.pos(pos_oh.to(first))
        x = self.embed_drop(x)
        for block in self.blocks:
            x = block.forward(x)
        if not same_device(x.device, self.output_device):
            x = x.to(self.output_device)
        x = self.ln_f(x)
        logits = self.head(x.reshape([x.shape[0] * x.shape[1], x.shape[2]]))
        return logits.reshape([x.shape[0], x.shape[1], self.vocab])

    def forward_cached_step(self, tok_oh, pos_oh):
        first = self.shard_devices[0]
        x = self.token(tok_oh.to(first))
        if self.pos is not None:
            x = x + self.pos(pos_oh.to(first))
        for block in self.blocks:
            x = block.forward_cached(x, self.cached_tokens)
        if not same_device(x.device, self.output_device):
            x = x.to(self.output_device)
        x = self.ln_f(x)
        logits = self.head(x.reshape([1, x.shape[2]])).reshape([1, 1, self.vocab])
        self.cached_tokens += 1
        return logits


def make_language_model_dataset(
    ids: np.ndarray, context_len: int
) -> Tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for i in range(len(ids) - context_len):
        xs.append(ids[i : i + context_len])
        ys.append(ids[i + 1 : i + context_len + 1])
    return np.array(xs, dtype=np.int32), np.array(ys, dtype=np.int32)


def split_train_val(X: np.ndarray, Y: np.ndarray, val_ratio: float):
    total = X.shape[0]
    val_count = max(1, int(total * val_ratio)) if total > 2 else 1
    val_count = min(val_count, max(1, total - 1))
    train_count = max(1, total - val_count)
    return (X[:train_count], Y[:train_count]), (X[train_count:], Y[train_count:])


def make_one_hot(tokens: np.ndarray, vocab: int) -> np.ndarray:
    out = np.zeros((tokens.shape[0], tokens.shape[1], vocab), dtype=np.float32)
    for b in range(tokens.shape[0]):
        for t in range(tokens.shape[1]):
            out[b, t, tokens[b, t]] = 1.0
    return out


def make_pos(batch: int, ctx: int) -> np.ndarray:
    out = np.zeros((batch, ctx, ctx), dtype=np.float32)
    for b in range(batch):
        for t in range(ctx):
            out[b, t, t] = 1.0
    return out


def make_single_pos(index: int, ctx: int) -> np.ndarray:
    clipped = min(index, ctx - 1)
    out = np.zeros((1, 1, ctx), dtype=np.float32)
    out[0, 0, clipped] = 1.0
    return out


@dataclass
class BatchArrays:
    tokens: np.ndarray
    targets: np.ndarray
    positions: np.ndarray


def split_train_val_ids(ids: np.ndarray, context_len: int, val_ratio: float):
    if ids.shape[0] <= context_len + 2:
        raise ValueError("Corpus is too small for the requested context length")
    total_windows = ids.shape[0] - context_len
    val_windows = max(1, int(total_windows * val_ratio)) if total_windows > 2 else 1
    val_windows = min(val_windows, max(1, total_windows - 1))
    split = ids.shape[0] - val_windows
    split = max(context_len + 1, split)
    return ids[:split], ids[split - context_len :]


def sample_batch_from_ids(
    ids: np.ndarray,
    context_len: int,
    batch_size: int,
    vocab: int,
    starts: Optional[np.ndarray] = None,
) -> BatchArrays:
    max_start = ids.shape[0] - context_len - 1
    if max_start < 0:
        raise ValueError("Not enough tokens to sample a batch")
    if starts is None:
        starts = np.random.randint(0, max_start + 1, size=batch_size)
    x = np.stack([ids[s : s + context_len] for s in starts]).astype(np.int32)
    y = np.stack([ids[s + 1 : s + context_len + 1] for s in starts]).astype(np.int32)
    return BatchArrays(
        tokens=make_one_hot(x, vocab),
        targets=make_one_hot(y, vocab),
        positions=make_pos(x.shape[0], context_len),
    )


def iter_eval_batches(
    ids: np.ndarray, context_len: int, batch_size: int, vocab: int, max_batches: int
):
    max_start = ids.shape[0] - context_len - 1
    if max_start < 0:
        return
    starts = np.arange(max_start + 1)
    total_batches = min(max_batches, max(1, int(np.ceil(starts.shape[0] / batch_size))))
    for batch_idx in range(total_batches):
        begin = batch_idx * batch_size
        end = min(begin + batch_size, starts.shape[0])
        yield sample_batch_from_ids(
            ids, context_len, end - begin, vocab, starts=starts[begin:end]
        )


def prepare_prompt_window(
    tokenizer: CharacterTokenizer, prompt: str, ctx: int
) -> np.ndarray:
    ids = tokenizer.encode(prompt)
    if ids.shape[0] >= ctx:
        return ids[-ctx:]
    pad = np.full(
        (ctx - ids.shape[0],), tokenizer.stoi[tokenizer.pad_token], dtype=np.int32
    )
    return np.concatenate([pad, ids])


def resolve_sampling_config(args) -> Dict[str, float]:
    if getattr(args, "sampling_preset", "manual") != "manual":
        preset = SAMPLING_PRESETS[args.sampling_preset]
        return {
            "temperature": float(preset["temperature"]),
            "top_k": int(preset["top_k"]),
            "top_p": float(preset["top_p"]),
            "repetition_penalty": float(preset["repetition_penalty"]),
        }
    return {
        "temperature": float(args.temperature),
        "top_k": int(args.top_k),
        "top_p": float(args.top_p),
        "repetition_penalty": float(args.repetition_penalty),
    }


def sample_next_id(
    logits_np: np.ndarray,
    temperature: float,
    top_k: int,
    top_p: float,
    repetition_penalty: float,
    recent_token_ids: np.ndarray,
) -> int:
    adjusted = np.array(logits_np, copy=True)
    if repetition_penalty > 1.0 and recent_token_ids.size > 0:
        for token_id in np.unique(recent_token_ids):
            if adjusted[token_id] >= 0.0:
                adjusted[token_id] /= repetition_penalty
            else:
                adjusted[token_id] *= repetition_penalty
    scaled = adjusted / max(temperature, 1e-5)
    scaled = scaled - np.max(scaled)
    probs = np.exp(scaled)
    probs /= max(np.sum(probs), 1e-12)
    if top_k > 0 and top_k < probs.shape[0]:
        keep = np.argpartition(probs, -top_k)[-top_k:]
        masked = np.zeros_like(probs)
        masked[keep] = probs[keep]
        probs = masked / max(np.sum(masked), 1e-12)
    if 0.0 < top_p < 1.0:
        order = np.argsort(probs)[::-1]
        sorted_probs = probs[order]
        cumulative = np.cumsum(sorted_probs)
        keep_mask = cumulative <= top_p
        if keep_mask.size > 0:
            keep_mask[0] = True
        keep_indices = order[keep_mask]
        masked = np.zeros_like(probs)
        masked[keep_indices] = probs[keep_indices]
        probs = masked / max(np.sum(masked), 1e-12)
    return int(np.random.choice(np.arange(probs.shape[0]), p=probs))


def model_forward_logits(
    model: FullGPTDemoModel, token_ids: np.ndarray, tokenizer: CharacterTokenizer
):
    tok = make_one_hot(token_ids.reshape(1, -1), tokenizer.vocab_size)
    pos = make_pos(1, model.ctx)
    with munet.no_grad():
        logits = model.forward(munet.from_numpy(tok), munet.from_numpy(pos))
    cpu = munet.Device(munet.DeviceType.CPU, 0)
    return np.array(logits.to(cpu).detach(), copy=False)[0, -1]


def model_forward_logits_cached(
    model: FullGPTDemoModel, token_ids: np.ndarray, tokenizer: CharacterTokenizer
):
    window = token_ids[-model.ctx :]
    with munet.no_grad():
        model.reset_kv_cache()
        logits = None
        for index, token_id in enumerate(window):
            tok = make_one_hot(
                np.array([[token_id]], dtype=np.int32), tokenizer.vocab_size
            )
            pos = make_single_pos(index, model.ctx)
            logits = model.forward_cached_step(
                munet.from_numpy(tok), munet.from_numpy(pos)
            )
    cpu = munet.Device(munet.DeviceType.CPU, 0)
    return np.array(logits.to(cpu).detach(), copy=False)[0, -1]


def generate_text(
    model: FullGPTDemoModel,
    tokenizer: CharacterTokenizer,
    prompt: str,
    max_new_tokens: int,
    sampling_cfg: Dict[str, float],
    use_kv_cache: bool = True,
    return_stats: bool = False,
):
    prompt_ids = tokenizer.encode(prompt)
    if prompt_ids.size == 0:
        prompt_ids = np.array([tokenizer.stoi[tokenizer.pad_token]], dtype=np.int32)
    window = prompt_ids[-model.ctx :].copy()
    generated = list(tokenizer.sanitize(prompt))
    stats = {"cache_enabled": bool(use_kv_cache), "reprime_count": 0, "steps": 0}
    if use_kv_cache:
        logits_np = model_forward_logits_cached(model, window, tokenizer)
    else:
        window = prepare_prompt_window(tokenizer, prompt, model.ctx)
        logits_np = model_forward_logits(model, window, tokenizer)
    for _ in range(max_new_tokens):
        nxt = sample_next_id(
            logits_np,
            temperature=sampling_cfg["temperature"],
            top_k=sampling_cfg["top_k"],
            top_p=sampling_cfg["top_p"],
            repetition_penalty=sampling_cfg["repetition_penalty"],
            recent_token_ids=window,
        )
        generated.append(tokenizer.itos[nxt])
        stats["steps"] += 1
        window = np.concatenate([window, np.array([nxt], dtype=np.int32)])[-model.ctx :]
        if use_kv_cache:
            if model.cached_tokens >= model.ctx:
                stats["reprime_count"] += 1
                logits_np = model_forward_logits_cached(model, window, tokenizer)
            else:
                with munet.no_grad():
                    tok = make_one_hot(
                        np.array([[nxt]], dtype=np.int32), tokenizer.vocab_size
                    )
                    pos = make_single_pos(model.cached_tokens, model.ctx)
                    logits = model.forward_cached_step(
                        munet.from_numpy(tok), munet.from_numpy(pos)
                    )
                cpu = munet.Device(munet.DeviceType.CPU, 0)
                logits_np = np.array(logits.to(cpu).detach(), copy=False)[0, -1]
        else:
            logits_np = model_forward_logits(model, window, tokenizer)
    text = "".join(generated)
    return (text, stats) if return_stats else text


def tensor_to_numpy_copy(tensor):
    cpu = munet.Device(munet.DeviceType.CPU, 0)
    td = tensor.detach()
    if td.device.type != munet.DeviceType.CPU:
        td = td.to(cpu)
    return np.array(td, copy=False).copy()


def copy_numpy_into_existing_tensor(tensor, arr):
    req = bool(tensor.requires_grad)
    cpu_tensor = tensor.detach().to(munet.Device(munet.DeviceType.CPU, 0))
    np.array(cpu_tensor, copy=False)[:] = arr.astype(np.float32, copy=False)
    tensor.replace_(cpu_tensor.to(tensor.device))
    tensor.requires_grad = req


def apply_weight_decay(model, weight_decay: float):
    if weight_decay <= 0.0:
        return
    for name, param in model.named_parameters().items():
        if param.grad is None:
            continue
        if name.endswith("bias") or ".ln" in name or ".norm" in name:
            continue
        grad = tensor_to_numpy_copy(param.grad)
        grad += weight_decay * tensor_to_numpy_copy(param)
        copy_numpy_into_existing_tensor(param.grad, grad)


def clip_gradients(model, max_norm: float) -> float:
    total_sq = 0.0
    grads = []
    for _, param in model.named_parameters().items():
        if param.grad is None:
            continue
        grad = tensor_to_numpy_copy(param.grad)
        grads.append((param.grad, grad))
        total_sq += float(np.sum(grad * grad))
    total_norm = float(np.sqrt(total_sq))
    if max_norm > 0.0 and total_norm > max_norm and total_norm > 0.0:
        scale = max_norm / total_norm
        for grad_tensor, grad in grads:
            copy_numpy_into_existing_tensor(grad_tensor, grad * scale)
    return total_norm


def scheduled_lr(
    base_lr: float, step: int, total_steps: int, warmup_steps: int, min_lr_ratio: float
) -> float:
    warmup_steps = max(0, warmup_steps)
    if warmup_steps > 0 and step < warmup_steps:
        return base_lr * float(step + 1) / float(max(1, warmup_steps))
    decay_steps = max(1, total_steps - warmup_steps)
    progress = min(1.0, max(0.0, float(step - warmup_steps) / float(decay_steps)))
    cosine = 0.5 * (1.0 + np.cos(np.pi * progress))
    return base_lr * (min_lr_ratio + (1.0 - min_lr_ratio) * cosine)


def save_periodic_checkpoint(
    output_dir: str, model: FullGPTDemoModel, step: int, summary: Dict[str, float]
):
    ckpt_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    weight_path = os.path.join(ckpt_dir, f"checkpoint_step_{step:06d}.npz")
    meta_path = os.path.join(ckpt_dir, f"checkpoint_step_{step:06d}.json")
    save_parameter_state(model, weight_path)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return weight_path


def latest_checkpoint(model_dir: str) -> Optional[str]:
    ckpt_dir = os.path.join(model_dir, "checkpoints")
    if not os.path.isdir(ckpt_dir):
        return None
    candidates = sorted(name for name in os.listdir(ckpt_dir) if name.endswith(".npz"))
    if not candidates:
        return None
    return os.path.join(ckpt_dir, candidates[-1])


def print_runtime_guidance(ctx: int, d_model: int, shards: int):
    approx_tokens_per_forward = ctx * d_model
    print(
        "Runtime guidance: prefer smaller batch sizes on CPU, use more shards only when accelerators are available, "
        f"and expect activation cost to grow roughly with context*d_model ({approx_tokens_per_forward})."
    )
    print(
        f"For longer runs, checkpoint frequently, keep context <= {ctx}, and test with shards={shards} before scaling up."
    )


def save_parameter_state(model, path: str):
    state = {}
    for name, tensor in model.named_parameters().items():
        state[name] = tensor_to_numpy_copy(tensor)
    np.savez(path, **state)


def load_parameter_state(model, path: str):
    named = model.named_parameters()
    with np.load(path, allow_pickle=False) as state:
        for name, tensor in named.items():
            if name not in state:
                raise KeyError(f"Missing parameter '{name}' in saved state")
            cpu_tensor = tensor.to(munet.Device(munet.DeviceType.CPU, 0))
            munet.copy_from_numpy(cpu_tensor, state[name])
            tensor.replace_(cpu_tensor.to(tensor.device))


def evaluate_model(
    model, eval_ids: np.ndarray, batch_size: int, vocab_size: int, max_batches: int = 4
):
    if eval_ids.shape[0] <= model.ctx + 1:
        return float("nan"), float("nan")
    model.eval()
    losses = []
    for batch in iter_eval_batches(
        eval_ids, model.ctx, batch_size, vocab_size, max_batches
    ):
        xb = munet.from_numpy(batch.tokens)
        pb = munet.from_numpy(batch.positions)
        yb = (
            munet.from_numpy(batch.targets)
            .reshape([batch.tokens.shape[0] * model.ctx, vocab_size])
            .to(model.output_device)
        )
        with munet.no_grad():
            logits = model.forward(xb, pb).reshape(
                [batch.tokens.shape[0] * model.ctx, vocab_size]
            )
            loss = logits.cross_entropy(yb)
        losses.append(loss.item())
    mean_loss = float(np.mean(losses))
    return mean_loss, float(np.exp(mean_loss))


def read_corpus(args) -> str:
    if args.corpus_file:
        with open(args.corpus_file, "r", encoding="utf-8") as f:
            return f.read()
    return DEFAULT_CORPUS


def save_artifact(
    output_dir: str,
    model: FullGPTDemoModel,
    tokenizer: CharacterTokenizer,
    training_text: str,
    system_prompt: str,
    training_summary: Dict[str, float],
):
    os.makedirs(output_dir, exist_ok=True)
    weights_path = os.path.join(output_dir, "model_weights.npz")
    config_path = os.path.join(output_dir, "config.json")
    metrics_path = os.path.join(output_dir, "training_metrics.json")
    save_parameter_state(model, weights_path)
    config = {
        "ctx": model.ctx,
        "d_model": model.d_model,
        "n_heads": model.n_heads,
        "n_layers": model.n_layers,
        "ffn_type": model.ffn_type,
        "attn_dropout": model.attn_dropout,
        "resid_dropout": model.resid_dropout,
        "embed_dropout": model.embed_dropout,
        "norm_type": model.norm_type,
        "position_type": model.position_type,
        "chars": tokenizer.chars,
        "generation_uses_kv_cache": True,
        "training_text": training_text,
        "system_prompt": system_prompt,
        "default_checkpoint": (
            "best"
            if os.path.exists(os.path.join(output_dir, "best_model_weights.npz"))
            else "last"
        ),
    }
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(training_summary, f, indent=2)
    print(f"Saved weights to {weights_path}")
    print(f"Saved config to {config_path}")
    print(f"Saved metrics to {metrics_path}")


def checkpoint_path(model_dir: str, checkpoint: str) -> str:
    best = os.path.join(model_dir, "best_model_weights.npz")
    last = os.path.join(model_dir, "model_weights.npz")
    if checkpoint == "best" and os.path.exists(best):
        return best
    return last


def load_artifact(model_dir: str, requested_shards: int, checkpoint: str):
    config_path = os.path.join(model_dir, "config.json")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    tokenizer = CharacterTokenizer(cfg["chars"])
    shard_devices = resolve_shard_devices(requested_shards)
    model = FullGPTDemoModel(
        vocab=tokenizer.vocab_size,
        ctx=cfg["ctx"],
        d_model=cfg["d_model"],
        n_heads=cfg["n_heads"],
        n_layers=cfg["n_layers"],
        ffn_type=cfg.get("ffn_type", "gelu"),
        shard_devices=shard_devices,
        attn_dropout=cfg.get("attn_dropout", 0.0),
        resid_dropout=cfg.get("resid_dropout", 0.0),
        embed_dropout=cfg.get("embed_dropout", 0.0),
        norm_type=cfg.get("norm_type", "layernorm"),
        position_type=cfg.get("position_type", "learned"),
    )
    load_parameter_state(model, checkpoint_path(model_dir, checkpoint))
    model.eval()
    metrics_path = os.path.join(model_dir, "training_metrics.json")
    metrics = {}
    if os.path.exists(metrics_path):
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)
    return model, tokenizer, cfg, metrics


def train_command(args):
    training_text = read_corpus(args)
    tokenizer = CharacterTokenizer.from_text(training_text)
    shard_devices = resolve_shard_devices(args.shards)
    ids = tokenizer.encode(training_text)
    train_ids, val_ids = split_train_val_ids(ids, args.context, args.val_ratio)
    model = FullGPTDemoModel(
        vocab=tokenizer.vocab_size,
        ctx=args.context,
        d_model=args.d_model,
        n_heads=args.heads,
        n_layers=args.layers,
        ffn_type=args.ffn,
        shard_devices=shard_devices,
        attn_dropout=args.attn_dropout,
        resid_dropout=args.resid_dropout,
        embed_dropout=args.embed_dropout,
        norm_type=args.norm,
        position_type=args.positions,
    )
    opt = munet.optim.Adam(model.parameters(), lr=args.lr)
    sampling_cfg = resolve_sampling_config(args)
    print("Training shards:", ", ".join(repr(dev) for dev in shard_devices))
    train_windows = max(0, train_ids.shape[0] - args.context)
    val_windows = max(0, val_ids.shape[0] - args.context)
    print(
        f"Train windows={train_windows} | Val windows={val_windows} | vocab={tokenizer.vocab_size} | "
        f"ffn={args.ffn} | norm={args.norm} | positions={args.positions} | "
        f"attn_dropout={args.attn_dropout} | resid_dropout={args.resid_dropout}"
    )
    print(f"Sampling config: {sampling_cfg}")
    print_runtime_guidance(args.context, args.d_model, args.shards)
    best_val_loss = float("inf")
    best_step = -1
    last_train_loss = float("nan")
    last_train_ppl = float("nan")
    resumed_from = None
    if args.resume:
        resume_path = (
            latest_checkpoint(args.output_dir)
            if args.resume == "latest"
            else args.resume
        )
        if resume_path:
            load_parameter_state(model, resume_path)
            resumed_from = resume_path
            print(f"Resumed weights from {resume_path}")
    for step in range(args.steps):
        model.train()
        batch = sample_batch_from_ids(
            train_ids, args.context, args.batch_size, tokenizer.vocab_size
        )
        xb = munet.from_numpy(batch.tokens)
        pb = munet.from_numpy(batch.positions)
        yb = (
            munet.from_numpy(batch.targets)
            .reshape([args.batch_size * args.context, tokenizer.vocab_size])
            .to(model.output_device)
        )
        current_lr = scheduled_lr(
            args.lr, step, args.steps, args.warmup_steps, args.min_lr_ratio
        )
        opt.lr = current_lr
        opt.zero_grad()
        logits = model.forward(xb, pb).reshape(
            [args.batch_size * args.context, tokenizer.vocab_size]
        )
        loss = logits.cross_entropy(yb)
        loss.backward()
        grad_norm = opt.clip_grad_norm(args.grad_clip)
        opt.step()
        opt.apply_weight_decay(args.weight_decay)
        last_train_loss = loss.item()
        last_train_ppl = loss.exp().to(munet.Device(munet.DeviceType.CPU, 0)).item()
        if args.checkpoint_every > 0 and ((step + 1) % args.checkpoint_every == 0):
            save_periodic_checkpoint(
                args.output_dir,
                model,
                step + 1,
                {"step": step + 1, "lr": current_lr, "grad_norm": grad_norm},
            )
        if step % args.eval_every == 0 or step == args.steps - 1:
            val_loss, val_ppl = evaluate_model(
                model, val_ids, args.batch_size, tokenizer.vocab_size, args.eval_batches
            )
            improved = val_loss < best_val_loss
            if improved:
                best_val_loss = val_loss
                best_step = step
                os.makedirs(args.output_dir, exist_ok=True)
                save_parameter_state(
                    model, os.path.join(args.output_dir, "best_model_weights.npz")
                )
            print(
                f"step {step:04d} | lr={current_lr:.6f} | grad_norm={grad_norm:.4f} | train_loss={last_train_loss:.4f} | train_ppl={last_train_ppl:.4f} | "
                f"val_loss={val_loss:.4f} | val_ppl={val_ppl:.4f} | best_step={best_step}"
            )
    model.eval()
    summary = {
        "last_train_loss": last_train_loss,
        "last_train_ppl": last_train_ppl,
        "best_val_loss": best_val_loss,
        "best_val_ppl": (
            float(np.exp(best_val_loss)) if np.isfinite(best_val_loss) else float("nan")
        ),
        "best_step": best_step,
        "sampling_config": sampling_cfg,
        "val_ratio": args.val_ratio,
        "grad_clip": args.grad_clip,
        "weight_decay": args.weight_decay,
        "warmup_steps": args.warmup_steps,
        "min_lr_ratio": args.min_lr_ratio,
        "checkpoint_every": args.checkpoint_every,
        "resumed_from": resumed_from,
    }
    save_artifact(
        args.output_dir, model, tokenizer, training_text, args.system_prompt, summary
    )
    if best_step >= 0:
        shutil.copyfile(
            os.path.join(args.output_dir, "best_model_weights.npz"),
            os.path.join(args.output_dir, "selected_model_weights.npz"),
        )
    preview_prompt = f"System: {args.system_prompt}\nUser: hello\nAssistant:"
    preview = generate_text(
        model, tokenizer, preview_prompt, args.generate, sampling_cfg
    )
    print("\nPreview generation:\n" + preview)


def generate_command(args):
    model, tokenizer, cfg, metrics = load_artifact(
        args.model_dir, args.shards, args.checkpoint
    )
    set_seed(args.seed)
    sampling_cfg = resolve_sampling_config(args)
    print(
        "Loaded model with shards:", ", ".join(repr(dev) for dev in model.shard_devices)
    )
    print(f"Sampling config: {sampling_cfg}")
    start = time.time()
    text, stats = generate_text(
        model,
        tokenizer,
        args.prompt,
        args.generate,
        sampling_cfg,
        use_kv_cache=not args.disable_kv_cache,
        return_stats=True,
    )
    elapsed = max(time.time() - start, 1e-9)
    print(text)
    print(
        f"\nInference stats: cache={stats['cache_enabled']} | reprimes={stats['reprime_count']} | tokens/s={stats['steps'] / elapsed:.2f}"
    )
    if args.show_training_info:
        print("\nTraining corpus length:", len(cfg.get("training_text", "")))
        if metrics:
            print("Training metrics:", json.dumps(metrics, indent=2))
        print_runtime_guidance(
            cfg.get("ctx", model.ctx), cfg.get("d_model", model.d_model), args.shards
        )


def chat_command(args):
    model, tokenizer, cfg, metrics = load_artifact(
        args.model_dir, args.shards, args.checkpoint
    )
    set_seed(args.seed)
    sampling_cfg = resolve_sampling_config(args)
    system_prompt = cfg.get("system_prompt", DEFAULT_SYSTEM_PROMPT)
    history = [f"System: {system_prompt}"]
    print(
        "Loaded chat model with shards:",
        ", ".join(repr(dev) for dev in model.shard_devices),
    )
    print(f"Sampling config: {sampling_cfg}")
    if metrics:
        print(f"Best validation perplexity: {metrics.get('best_val_ppl', 'n/a')}")
    print_runtime_guidance(
        cfg.get("ctx", model.ctx), cfg.get("d_model", model.d_model), args.shards
    )
    print("Enter messages. Type /quit to exit.\n")
    while True:
        try:
            user = input("user> ")
        except EOFError:
            print()
            break
        if user.strip().lower() in {"/quit", "/exit"}:
            break
        history.append(f"User: {user}")
        prompt = "\n".join(history + ["Assistant:"])
        start = time.time()
        response_full, stats = generate_text(
            model,
            tokenizer,
            prompt,
            args.generate,
            sampling_cfg,
            use_kv_cache=not args.disable_kv_cache,
            return_stats=True,
        )
        elapsed = max(time.time() - start, 1e-9)
        response = response_full[len(prompt) :]
        for marker in ["\nUser:", "\nSystem:"]:
            if marker in response:
                response = response.split(marker, 1)[0]
        response = (
            response.strip()
            or "I need a little more training data to answer that well."
        )
        print(f"assistant> {response}")
        print(
            f"[cache={stats['cache_enabled']} reprimes={stats['reprime_count']} tokens/s={stats['steps'] / elapsed:.2f}]\n"
        )
        history.append(f"Assistant: {response}")


def infer_command(args):
    if args.mode == "generate":
        generate_command(args)
        return
    chat_command(args)


def add_sampling_args(parser, default_preset: str = "balanced"):
    parser.add_argument(
        "--sampling-preset",
        choices=["manual", *SAMPLING_PRESETS.keys()],
        default=default_preset,
    )
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--top-p", type=float, default=0.92)
    parser.add_argument("--repetition-penalty", type=float, default=1.08)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Train, save/load, generate, and chat with a tiny GPT-3-style MuNet model."
    )
    sub = parser.add_subparsers(dest="command", required=True)
    train = sub.add_parser(
        "train", help="Train a tiny GPT-style model and save weights/config."
    )
    train.add_argument("--output-dir", type=str, default="/tmp/munet_gpt3_demo")
    train.add_argument("--corpus-file", type=str, default=None)
    train.add_argument("--system-prompt", type=str, default=DEFAULT_SYSTEM_PROMPT)
    train.add_argument("--context", type=int, default=32)
    train.add_argument("--d-model", type=int, default=64)
    train.add_argument("--heads", type=int, default=4)
    train.add_argument("--layers", type=int, default=4)
    train.add_argument("--batch-size", type=int, default=8)
    train.add_argument("--steps", type=int, default=120)
    train.add_argument("--lr", type=float, default=3e-3)
    train.add_argument("--shards", type=int, default=2)
    train.add_argument("--ffn", choices=["gelu", "swiglu"], default="gelu")
    train.add_argument("--norm", choices=["layernorm", "rmsnorm"], default="layernorm")
    train.add_argument("--positions", choices=["learned", "rope"], default="learned")
    train.add_argument("--attn-dropout", type=float, default=0.1)
    train.add_argument("--resid-dropout", type=float, default=0.1)
    train.add_argument("--embed-dropout", type=float, default=0.05)
    train.add_argument("--val-ratio", type=float, default=0.1)
    train.add_argument("--eval-every", type=int, default=20)
    train.add_argument("--eval-batches", type=int, default=4)
    train.add_argument("--generate", type=int, default=96)
    train.add_argument("--grad-clip", type=float, default=1.0)
    train.add_argument("--weight-decay", type=float, default=0.0)
    train.add_argument("--warmup-steps", type=int, default=10)
    train.add_argument("--min-lr-ratio", type=float, default=0.1)
    train.add_argument("--checkpoint-every", type=int, default=50)
    train.add_argument(
        "--resume",
        type=str,
        default=None,
        help='Path to a checkpoint .npz or "latest" to resume from the newest saved checkpoint.',
    )
    train.add_argument("--seed", type=int, default=1337)
    add_sampling_args(train)
    train.set_defaults(func=train_command)
    generate = sub.add_parser(
        "generate", help="Load a trained model and continue a prompt."
    )
    generate.add_argument("--model-dir", type=str, required=True)
    generate.add_argument("--prompt", type=str, required=True)
    generate.add_argument("--generate", type=int, default=96)
    generate.add_argument("--shards", type=int, default=2)
    generate.add_argument("--checkpoint", choices=["best", "last"], default="best")
    generate.add_argument("--show-training-info", action="store_true")
    generate.add_argument("--seed", type=int, default=1337)
    generate.add_argument("--disable-kv-cache", action="store_true")
    add_sampling_args(generate)
    generate.set_defaults(func=generate_command)
    chat = sub.add_parser("chat", help="Load a trained model and enter a chat loop.")
    chat.add_argument("--model-dir", type=str, required=True)
    chat.add_argument("--generate", type=int, default=120)
    chat.add_argument("--shards", type=int, default=2)
    chat.add_argument("--checkpoint", choices=["best", "last"], default="best")
    chat.add_argument("--seed", type=int, default=1337)
    chat.add_argument("--disable-kv-cache", action="store_true")
    add_sampling_args(chat)
    chat.set_defaults(func=chat_command)
    infer = sub.add_parser(
        "infer",
        help="Reproducible inference entrypoint for generation or chat with KV-cache enabled by default.",
    )
    infer.add_argument("--model-dir", type=str, required=True)
    infer.add_argument("--mode", choices=["generate", "chat"], default="chat")
    infer.add_argument("--prompt", type=str, default="")
    infer.add_argument("--generate", type=int, default=120)
    infer.add_argument("--shards", type=int, default=2)
    infer.add_argument("--checkpoint", choices=["best", "last"], default="best")
    infer.add_argument("--show-training-info", action="store_true")
    infer.add_argument("--seed", type=int, default=1337)
    infer.add_argument("--disable-kv-cache", action="store_true")
    add_sampling_args(infer)
    infer.set_defaults(func=infer_command)
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    if hasattr(args, "seed"):
        set_seed(args.seed)
    args.func(args)


if __name__ == "__main__":
    main()
