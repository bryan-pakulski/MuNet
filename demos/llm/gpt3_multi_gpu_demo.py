import argparse
import json
import os
import shutil
import sys
from typing import Dict, List, Tuple

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
    "deterministic": {"temperature": 0.2, "top_k": 1, "top_p": 1.0, "repetition_penalty": 1.02},
    "balanced": {"temperature": 0.8, "top_k": 8, "top_p": 0.92, "repetition_penalty": 1.08},
    "creative": {"temperature": 1.0, "top_k": 16, "top_p": 0.96, "repetition_penalty": 1.12},
}


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
        expanded = [accelerators[i % len(accelerators)] for i in range(requested_shards)]
        print(
            f"Only found {len(accelerators)} accelerator(s); reusing them for {requested_shards} shard(s): "
            + ", ".join(repr(dev) for dev in expanded)
        )
        return expanded

    cpu = munet.Device(munet.DeviceType.CPU, 0)
    fallback = [cpu for _ in range(requested_shards)]
    print("No accelerators detected; running the sharded GPT demo on CPU fallback shards.")
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
    def __init__(self, d_model: int, n_heads: int, device, use_rope: bool, attn_dropout: float):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        opts = munet.TensorOptions()
        opts.device = device
        self.device = device
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.use_rope = use_rope
        self.q_proj = munet.nn.Linear(d_model, d_model, True, options=opts)
        self.k_proj = munet.nn.Linear(d_model, d_model, True, options=opts)
        self.v_proj = munet.nn.Linear(d_model, d_model, True, options=opts)
        self.out_proj = munet.nn.Linear(d_model, d_model, True, options=opts)
        self.attn_drop = munet.nn.Dropout(attn_dropout)
        self._rope_cache = {}

    def _rope_tables(self, seq_len: int, device):
        key = (seq_len, device.type, device.index)
        if key in self._rope_cache:
            return self._rope_cache[key]
        half = self.head_dim // 2
        base = 10000.0
        inv_freq = 1.0 / (base ** (np.arange(half, dtype=np.float32) / max(1, half)))
        positions = np.arange(seq_len, dtype=np.float32)[:, None]
        angles = positions * inv_freq[None, :]
        cos = munet.from_numpy(np.cos(angles).reshape(1, 1, seq_len, half).astype(np.float32)).to(device)
        sin = munet.from_numpy(np.sin(angles).reshape(1, 1, seq_len, half).astype(np.float32)).to(device)
        self._rope_cache[key] = (cos, sin)
        return cos, sin

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

    def forward(self, x):
        bsz, seq_len, _ = x.shape
        flat = x.reshape([bsz * seq_len, self.d_model])
        q = self.q_proj(flat).reshape([bsz, seq_len, self.n_heads, self.head_dim]).permute([0, 2, 1, 3]).contiguous()
        k = self.k_proj(flat).reshape([bsz, seq_len, self.n_heads, self.head_dim]).permute([0, 2, 1, 3]).contiguous()
        v = self.v_proj(flat).reshape([bsz, seq_len, self.n_heads, self.head_dim]).permute([0, 2, 1, 3]).contiguous()

        q = self._apply_rope(q)
        k = self._apply_rope(k)

        q2 = q.reshape([bsz * self.n_heads * seq_len, self.head_dim])
        k2 = k.reshape([bsz * self.n_heads * seq_len, self.head_dim])
        v2 = v.reshape([bsz * self.n_heads * seq_len, self.head_dim])
        scores = q2 @ k2.transpose(0, 1)
        scores = scores * (1.0 / np.sqrt(float(self.head_dim)))

        cpu = munet.Device(munet.DeviceType.CPU, 0)
        mask_cpu = munet.Tensor([bsz * self.n_heads * seq_len, bsz * self.n_heads * seq_len], device=cpu, dtype=munet.DataType.Int32)
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
        merged = ctx.reshape([bsz, self.n_heads, seq_len, self.head_dim]).permute([0, 2, 1, 3]).contiguous().reshape([bsz * seq_len, self.d_model])
        return self.out_proj(merged).reshape([bsz, seq_len, self.d_model])


class GPTDecoderBlock(munet.nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        ff_hidden: int,
        ffn_type: str,
        norm_type: str,
        position_type: str,
        device,
        attn_dropout: float,
        resid_dropout: float,
    ):
        super().__init__()
        opts = munet.TensorOptions()
        opts.device = device
        self.shard_device = device
        self.ln1 = build_norm(norm_type, d_model, opts)
        self.attn = RotarySelfAttention(d_model, n_heads, device, position_type == "rope", attn_dropout)
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
        self.pos = munet.nn.Embedding(ctx, d_model, options=first_opts) if position_type == "learned" else None
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


def make_language_model_dataset(ids: np.ndarray, context_len: int) -> Tuple[np.ndarray, np.ndarray]:
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


def prepare_prompt_window(tokenizer: CharacterTokenizer, prompt: str, ctx: int) -> np.ndarray:
    ids = tokenizer.encode(prompt)
    if ids.shape[0] >= ctx:
        return ids[-ctx:]
    pad = np.full((ctx - ids.shape[0],), tokenizer.stoi[tokenizer.pad_token], dtype=np.int32)
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


def model_forward_logits(model: FullGPTDemoModel, token_ids: np.ndarray, tokenizer: CharacterTokenizer):
    tok = make_one_hot(token_ids.reshape(1, -1), tokenizer.vocab_size)
    pos = make_pos(1, model.ctx)
    with munet.no_grad():
        logits = model.forward(munet.from_numpy(tok), munet.from_numpy(pos))
    cpu = munet.Device(munet.DeviceType.CPU, 0)
    return np.array(logits.to(cpu).detach(), copy=False)[0, -1]


def generate_text(model: FullGPTDemoModel, tokenizer: CharacterTokenizer, prompt: str,
                  max_new_tokens: int, sampling_cfg: Dict[str, float]) -> str:
    window = prepare_prompt_window(tokenizer, prompt, model.ctx)
    generated = list(tokenizer.sanitize(prompt))

    for _ in range(max_new_tokens):
        logits_np = model_forward_logits(model, window, tokenizer)
        nxt = sample_next_id(
            logits_np,
            temperature=sampling_cfg["temperature"],
            top_k=sampling_cfg["top_k"],
            top_p=sampling_cfg["top_p"],
            repetition_penalty=sampling_cfg["repetition_penalty"],
            recent_token_ids=window,
        )
        generated.append(tokenizer.itos[nxt])
        window = np.concatenate([window[1:], np.array([nxt], dtype=np.int32)])

    return "".join(generated)


def tensor_to_numpy_copy(tensor):
    cpu = munet.Device(munet.DeviceType.CPU, 0)
    td = tensor.detach()
    if td.device.type != munet.DeviceType.CPU:
        td = td.to(cpu)
    return np.array(td, copy=False).copy()


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


def evaluate_model(model, X_oh, Y_oh, P_oh, batch_size: int, vocab_size: int, max_batches: int = 4):
    if X_oh.shape[0] == 0:
        return float("nan"), float("nan")

    model.eval()
    losses = []
    batches = min(max_batches, max(1, int(np.ceil(X_oh.shape[0] / batch_size))))
    for batch_idx in range(batches):
        start = (batch_idx * batch_size) % X_oh.shape[0]
        end = min(start + batch_size, X_oh.shape[0])
        xb = munet.from_numpy(X_oh[start:end])
        pb = munet.from_numpy(P_oh[start:end])
        yb = munet.from_numpy(Y_oh[start:end]).reshape([(end - start) * model.ctx, vocab_size]).to(model.output_device)
        with munet.no_grad():
            logits = model.forward(xb, pb).reshape([(end - start) * model.ctx, vocab_size])
            loss = logits.cross_entropy(yb)
        losses.append(loss.item())

    mean_loss = float(np.mean(losses))
    return mean_loss, float(np.exp(mean_loss))


def read_corpus(args) -> str:
    if args.corpus_file:
        with open(args.corpus_file, "r", encoding="utf-8") as f:
            return f.read()
    return DEFAULT_CORPUS


def save_artifact(output_dir: str, model: FullGPTDemoModel, tokenizer: CharacterTokenizer,
                  training_text: str, system_prompt: str, training_summary: Dict[str, float]):
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
        "training_text": training_text,
        "system_prompt": system_prompt,
        "default_checkpoint": "best" if os.path.exists(os.path.join(output_dir, "best_model_weights.npz")) else "last",
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
    X_all, Y_all = make_language_model_dataset(ids, args.context)
    (X_train, Y_train), (X_val, Y_val) = split_train_val(X_all, Y_all, args.val_ratio)

    X_train_oh = make_one_hot(X_train, tokenizer.vocab_size)
    Y_train_oh = make_one_hot(Y_train, tokenizer.vocab_size)
    P_train = make_pos(X_train.shape[0], args.context)

    X_val_oh = make_one_hot(X_val, tokenizer.vocab_size)
    Y_val_oh = make_one_hot(Y_val, tokenizer.vocab_size)
    P_val = make_pos(X_val.shape[0], args.context)

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
    print(
        f"Train windows={X_train.shape[0]} | Val windows={X_val.shape[0]} | vocab={tokenizer.vocab_size} | "
        f"ffn={args.ffn} | norm={args.norm} | positions={args.positions} | "
        f"attn_dropout={args.attn_dropout} | resid_dropout={args.resid_dropout}"
    )
    print(f"Sampling config: {sampling_cfg}")

    best_val_loss = float("inf")
    best_step = -1
    last_train_loss = float("nan")
    last_train_ppl = float("nan")

    for step in range(args.steps):
        model.train()
        idx = np.random.randint(0, X_train.shape[0], size=args.batch_size)
        xb = munet.from_numpy(X_train_oh[idx])
        pb = munet.from_numpy(P_train[idx])
        yb = munet.from_numpy(Y_train_oh[idx]).reshape([args.batch_size * args.context, tokenizer.vocab_size]).to(
            model.output_device
        )

        opt.zero_grad()
        logits = model.forward(xb, pb).reshape([args.batch_size * args.context, tokenizer.vocab_size])
        loss = logits.cross_entropy(yb)
        loss.backward()
        opt.step()

        last_train_loss = loss.item()
        last_train_ppl = loss.exp().to(munet.Device(munet.DeviceType.CPU, 0)).item()

        if step % args.eval_every == 0 or step == args.steps - 1:
            val_loss, val_ppl = evaluate_model(
                model, X_val_oh, Y_val_oh, P_val, args.batch_size, tokenizer.vocab_size, args.eval_batches
            )
            improved = val_loss < best_val_loss
            if improved:
                best_val_loss = val_loss
                best_step = step
                os.makedirs(args.output_dir, exist_ok=True)
                save_parameter_state(model, os.path.join(args.output_dir, "best_model_weights.npz"))

            print(
                f"step {step:04d} | train_loss={last_train_loss:.4f} | train_ppl={last_train_ppl:.4f} | "
                f"val_loss={val_loss:.4f} | val_ppl={val_ppl:.4f} | best_step={best_step}"
            )

    model.eval()
    summary = {
        "last_train_loss": last_train_loss,
        "last_train_ppl": last_train_ppl,
        "best_val_loss": best_val_loss,
        "best_val_ppl": float(np.exp(best_val_loss)) if np.isfinite(best_val_loss) else float("nan"),
        "best_step": best_step,
        "sampling_config": sampling_cfg,
        "val_ratio": args.val_ratio,
    }
    save_artifact(args.output_dir, model, tokenizer, training_text, args.system_prompt, summary)
    if best_step >= 0:
        shutil.copyfile(os.path.join(args.output_dir, "best_model_weights.npz"), os.path.join(args.output_dir, "selected_model_weights.npz"))

    preview_prompt = f"System: {args.system_prompt}\nUser: hello\nAssistant:"
    preview = generate_text(model, tokenizer, preview_prompt, args.generate, sampling_cfg)
    print("\nPreview generation:\n" + preview)


def generate_command(args):
    model, tokenizer, cfg, metrics = load_artifact(args.model_dir, args.shards, args.checkpoint)
    sampling_cfg = resolve_sampling_config(args)
    print("Loaded model with shards:", ", ".join(repr(dev) for dev in model.shard_devices))
    print(f"Sampling config: {sampling_cfg}")
    text = generate_text(model, tokenizer, args.prompt, args.generate, sampling_cfg)
    print(text)
    if args.show_training_info:
        print("\nTraining corpus length:", len(cfg.get("training_text", "")))
        if metrics:
            print("Training metrics:", json.dumps(metrics, indent=2))


def chat_command(args):
    model, tokenizer, cfg, metrics = load_artifact(args.model_dir, args.shards, args.checkpoint)
    sampling_cfg = resolve_sampling_config(args)
    system_prompt = cfg.get("system_prompt", DEFAULT_SYSTEM_PROMPT)
    history = [f"System: {system_prompt}"]
    print("Loaded chat model with shards:", ", ".join(repr(dev) for dev in model.shard_devices))
    print(f"Sampling config: {sampling_cfg}")
    if metrics:
        print(f"Best validation perplexity: {metrics.get('best_val_ppl', 'n/a')}")
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
        response_full = generate_text(model, tokenizer, prompt, args.generate, sampling_cfg)
        response = response_full[len(prompt):]
        for marker in ["\nUser:", "\nSystem:"]:
            if marker in response:
                response = response.split(marker, 1)[0]
        response = response.strip() or "I need a little more training data to answer that well."
        print(f"assistant> {response}\n")
        history.append(f"Assistant: {response}")


def add_sampling_args(parser, default_preset: str = "balanced"):
    parser.add_argument("--sampling-preset", choices=["manual", *SAMPLING_PRESETS.keys()], default=default_preset)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--top-p", type=float, default=0.92)
    parser.add_argument("--repetition-penalty", type=float, default=1.08)


def build_parser():
    parser = argparse.ArgumentParser(description="Train, save/load, generate, and chat with a tiny GPT-3-style MuNet model.")
    sub = parser.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train", help="Train a tiny GPT-style model and save weights/config.")
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
    add_sampling_args(train)
    train.set_defaults(func=train_command)

    generate = sub.add_parser("generate", help="Load a trained model and continue a prompt.")
    generate.add_argument("--model-dir", type=str, required=True)
    generate.add_argument("--prompt", type=str, required=True)
    generate.add_argument("--generate", type=int, default=96)
    generate.add_argument("--shards", type=int, default=2)
    generate.add_argument("--checkpoint", choices=["best", "last"], default="best")
    generate.add_argument("--show-training-info", action="store_true")
    add_sampling_args(generate)
    generate.set_defaults(func=generate_command)

    chat = sub.add_parser("chat", help="Load a trained model and enter a chat loop.")
    chat.add_argument("--model-dir", type=str, required=True)
    chat.add_argument("--generate", type=int, default=120)
    chat.add_argument("--shards", type=int, default=2)
    chat.add_argument("--checkpoint", choices=["best", "last"], default="best")
    add_sampling_args(chat)
    chat.set_defaults(func=chat_command)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
