import argparse
import json
import os
import sys
from typing import List, Tuple

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
        chars = sorted(set(text))
        return cls(chars)

    @property
    def vocab_size(self) -> int:
        return len(self.chars)

    def sanitize(self, text: str) -> str:
        return "".join(ch if ch in self.stoi else self.pad_token for ch in text)

    def encode(self, text: str) -> np.ndarray:
        clean = self.sanitize(text)
        return np.array([self.stoi[ch] for ch in clean], dtype=np.int32)

    def decode(self, ids: np.ndarray) -> str:
        return "".join(self.itos[int(idx)] for idx in ids)


class FeedForwardGELU(munet.nn.Module):
    def __init__(self, d_model: int, hidden: int, options):
        super().__init__()
        self.fc1 = munet.nn.Linear(d_model, hidden, True, options=options)
        self.act = munet.nn.GELU()
        self.fc2 = munet.nn.Linear(hidden, d_model, True, options=options)

    def forward(self, x):
        h = self.fc1(x)
        return self.fc2(self.act(h))


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


class GPTDecoderBlock(munet.nn.Module):
    def __init__(self, d_model: int, n_heads: int, ff_hidden: int, ffn_type: str, device):
        super().__init__()
        opts = munet.TensorOptions()
        opts.device = device
        self.shard_device = device
        self.ln1 = munet.nn.LayerNorm(d_model, options=opts)
        self.attn = munet.nn.MultiHeadAttention(d_model, n_heads, causal=True, options=opts)
        self.ln2 = munet.nn.LayerNorm(d_model, options=opts)
        if ffn_type == "swiglu":
            self.ff = FeedForwardSwiGLU(d_model, ff_hidden, opts)
        else:
            self.ff = FeedForwardGELU(d_model, ff_hidden, opts)

    def forward(self, x):
        if not same_device(x.device, self.shard_device):
            x = x.to(self.shard_device)
        h = self.ln1(x)
        x = x + self.attn(h)
        h = self.ln2(x)
        flat = h.reshape([h.shape[0] * h.shape[1], h.shape[2]])
        ff = self.ff(flat).reshape([h.shape[0], h.shape[1], h.shape[2]])
        return x + ff


class FullGPTDemoModel(munet.nn.Module):
    """A trainable GPT-3-style decoder LM with optional model sharding.

    Current architecture is intentionally close to classic GPT/GPT-3 blocks:
    token embedding + learned positional embedding + pre-norm causal attention +
    residual MLP stack. `--ffn swiglu` enables a more modern feed-forward block,
    while RMSNorm/RoPE are left as roadmap items because MuNet still needs more
    tensor primitives for a clean backend-accelerated implementation.
    """

    def __init__(self, vocab: int, ctx: int, d_model: int, n_heads: int, n_layers: int,
                 ffn_type: str, shard_devices: List["munet.Device"]):
        super().__init__()
        self.vocab = vocab
        self.ctx = ctx
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.ffn_type = ffn_type
        self.shard_devices = shard_devices

        first_opts = munet.TensorOptions()
        first_opts.device = shard_devices[0]
        last_opts = munet.TensorOptions()
        last_opts.device = shard_devices[-1]

        self.token = munet.nn.Embedding(vocab, d_model, options=first_opts)
        self.pos = munet.nn.Embedding(ctx, d_model, options=first_opts)
        self.blocks = []
        ff_hidden = 4 * d_model
        for layer in range(n_layers):
            device = shard_devices[layer % len(shard_devices)]
            block = GPTDecoderBlock(d_model, n_heads, ff_hidden, ffn_type, device)
            setattr(self, f"block_{layer}", block)
            self.blocks.append(block)
        self.ln_f = munet.nn.LayerNorm(d_model, options=last_opts)
        self.head = munet.nn.Linear(d_model, vocab, True, options=last_opts)

    @property
    def output_device(self):
        return self.shard_devices[-1]

    def forward(self, tok_oh, pos_oh):
        first = self.shard_devices[0]
        x = self.token(tok_oh.to(first)) + self.pos(pos_oh.to(first))
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


def sample_next_id(logits_np: np.ndarray, temperature: float, top_k: int) -> int:
    scaled = logits_np / max(temperature, 1e-5)
    scaled = scaled - np.max(scaled)
    probs = np.exp(scaled)
    probs /= np.sum(probs)

    if top_k > 0 and top_k < probs.shape[0]:
        keep = np.argpartition(probs, -top_k)[-top_k:]
        masked = np.zeros_like(probs)
        masked[keep] = probs[keep]
        probs = masked / np.sum(masked)

    return int(np.random.choice(np.arange(probs.shape[0]), p=probs))


def model_forward_logits(model: FullGPTDemoModel, token_ids: np.ndarray, tokenizer: CharacterTokenizer):
    tok = make_one_hot(token_ids.reshape(1, -1), tokenizer.vocab_size)
    pos = make_pos(1, model.ctx)
    with munet.no_grad():
        logits = model.forward(munet.from_numpy(tok), munet.from_numpy(pos))
    cpu = munet.Device(munet.DeviceType.CPU, 0)
    return np.array(logits.to(cpu).detach(), copy=False)[0, -1]


def generate_text(model: FullGPTDemoModel, tokenizer: CharacterTokenizer, prompt: str,
                  max_new_tokens: int, temperature: float, top_k: int) -> str:
    window = prepare_prompt_window(tokenizer, prompt, model.ctx)
    generated = list(tokenizer.sanitize(prompt))

    for _ in range(max_new_tokens):
        logits_np = model_forward_logits(model, window, tokenizer)
        nxt = sample_next_id(logits_np, temperature=temperature, top_k=top_k)
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
            restored = cpu_tensor.to(tensor.device)
            tensor.replace_(restored)

def read_corpus(args) -> str:
    if args.corpus_file:
        with open(args.corpus_file, "r", encoding="utf-8") as f:
            return f.read()
    return DEFAULT_CORPUS


def save_artifact(output_dir: str, model: FullGPTDemoModel, tokenizer: CharacterTokenizer,
                  training_text: str, system_prompt: str):
    os.makedirs(output_dir, exist_ok=True)
    weights_path = os.path.join(output_dir, "model_weights.npz")
    config_path = os.path.join(output_dir, "config.json")

    save_parameter_state(model, weights_path)
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "ctx": model.ctx,
                "d_model": model.d_model,
                "n_heads": model.n_heads,
                "n_layers": model.n_layers,
                "ffn_type": model.ffn_type,
                "chars": tokenizer.chars,
                "training_text": training_text,
                "system_prompt": system_prompt,
            },
            f,
            indent=2,
        )

    print(f"Saved weights to {weights_path}")
    print(f"Saved config to {config_path}")


def load_artifact(model_dir: str, requested_shards: int):
    config_path = os.path.join(model_dir, "config.json")
    weights_path = os.path.join(model_dir, "model_weights.npz")
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
    )
    load_parameter_state(model, weights_path)
    model.eval()
    return model, tokenizer, cfg


def train_command(args):
    training_text = read_corpus(args)
    tokenizer = CharacterTokenizer.from_text(training_text)
    shard_devices = resolve_shard_devices(args.shards)

    ids = tokenizer.encode(training_text)
    X, Y = make_language_model_dataset(ids, args.context)
    X_oh = make_one_hot(X, tokenizer.vocab_size)
    Y_oh = make_one_hot(Y, tokenizer.vocab_size)
    P_oh = make_pos(X.shape[0], args.context)

    model = FullGPTDemoModel(
        vocab=tokenizer.vocab_size,
        ctx=args.context,
        d_model=args.d_model,
        n_heads=args.heads,
        n_layers=args.layers,
        ffn_type=args.ffn,
        shard_devices=shard_devices,
    )
    opt = munet.optim.Adam(model.parameters(), lr=args.lr)

    print("Training shards:", ", ".join(repr(dev) for dev in shard_devices))
    print(f"Dataset windows={X.shape[0]} | vocab={tokenizer.vocab_size} | ffn={args.ffn}")

    for step in range(args.steps):
        idx = np.random.randint(0, X.shape[0], size=args.batch_size)
        xb = munet.from_numpy(X_oh[idx])
        pb = munet.from_numpy(P_oh[idx])
        yb = munet.from_numpy(Y_oh[idx]).reshape([args.batch_size * args.context, tokenizer.vocab_size]).to(
            model.output_device
        )

        opt.zero_grad()
        logits = model.forward(xb, pb).reshape([args.batch_size * args.context, tokenizer.vocab_size])
        loss = logits.cross_entropy(yb)
        loss.backward()
        opt.step()

        if step % args.log_every == 0 or step == args.steps - 1:
            ppl = loss.exp().to(munet.Device(munet.DeviceType.CPU, 0)).item()
            print(f"step {step:04d} | loss={loss.item():.4f} | ppl={ppl:.4f}")

    model.eval()
    save_artifact(args.output_dir, model, tokenizer, training_text, args.system_prompt)

    preview_prompt = f"System: {args.system_prompt}\nUser: hello\nAssistant:"
    preview = generate_text(model, tokenizer, preview_prompt, args.generate, args.temperature, args.top_k)
    print("\nPreview generation:\n" + preview)


def generate_command(args):
    model, tokenizer, cfg = load_artifact(args.model_dir, args.shards)
    print("Loaded model with shards:", ", ".join(repr(dev) for dev in model.shard_devices))
    text = generate_text(model, tokenizer, args.prompt, args.generate, args.temperature, args.top_k)
    print(text)
    if args.show_training_info:
        print("\nTraining corpus length:", len(cfg.get("training_text", "")))


def chat_command(args):
    model, tokenizer, cfg = load_artifact(args.model_dir, args.shards)
    system_prompt = cfg.get("system_prompt", DEFAULT_SYSTEM_PROMPT)
    history = [f"System: {system_prompt}"]
    print("Loaded chat model with shards:", ", ".join(repr(dev) for dev in model.shard_devices))
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
        response_full = generate_text(model, tokenizer, prompt, args.generate, args.temperature, args.top_k)
        response = response_full[len(prompt):]
        stop_markers = ["\nUser:", "\nSystem:"]
        for marker in stop_markers:
            if marker in response:
                response = response.split(marker, 1)[0]
        response = response.strip() or "I need a little more training data to answer that well."
        print(f"assistant> {response}\n")
        history.append(f"Assistant: {response}")


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
    train.add_argument("--log-every", type=int, default=20)
    train.add_argument("--generate", type=int, default=96)
    train.add_argument("--temperature", type=float, default=0.9)
    train.add_argument("--top-k", type=int, default=8)
    train.set_defaults(func=train_command)

    generate = sub.add_parser("generate", help="Load a trained model and continue a prompt.")
    generate.add_argument("--model-dir", type=str, required=True)
    generate.add_argument("--prompt", type=str, required=True)
    generate.add_argument("--generate", type=int, default=96)
    generate.add_argument("--temperature", type=float, default=0.8)
    generate.add_argument("--top-k", type=int, default=8)
    generate.add_argument("--shards", type=int, default=2)
    generate.add_argument("--show-training-info", action="store_true")
    generate.set_defaults(func=generate_command)

    chat = sub.add_parser("chat", help="Load a trained model and enter a chat loop.")
    chat.add_argument("--model-dir", type=str, required=True)
    chat.add_argument("--generate", type=int, default=120)
    chat.add_argument("--temperature", type=float, default=0.8)
    chat.add_argument("--top-k", type=int, default=8)
    chat.add_argument("--shards", type=int, default=2)
    chat.set_defaults(func=chat_command)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
