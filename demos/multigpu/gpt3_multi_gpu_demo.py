#!/usr/bin/env python3
"""Barebones multi-GPU GPT training/inference demo with per-device backend specification.

Usage:
    # Train on CUDA devices 0 and 1
    python gpt3_multi_gpu_demo.py train --devices "0:cuda,1:cuda"
    
    # Mixed backends - CUDA + Vulkan
    python gpt3_multi_gpu_demo.py train --devices "0:cuda,1:vulkan"
    
    # Generate with trained model
    python gpt3_multi_gpu_demo.py generate --devices "0:cuda" --model-dir ./model --prompt "Hello"
"""
import argparse
import os
import numpy as np

import munet
from munet import nn


def parse_device_spec(spec: str) -> list:
    """Parse device specification like "0:cuda,1:vulkan" into [munet.Device, ...].
    
    Supports formats:
        - "0:cuda,1:vulkan" -> [Device(CUDA, 0), Device(Vulkan, 1)]
        - "0,1" -> [Device(CUDA, 0), Device(CUDA, 1)] (defaults to CUDA)
        - "0:cuda" -> [Device(CUDA, 0)]
    """
    devices = []
    for part in spec.split(","):
        part = part.strip().strip('"').strip("'")
        if ":" in part:
            dev_id, backend = part.split(":", 1)
            backend = backend.capitalize()
        else:
            dev_id = part
            backend = "CUDA"
        devices.append((backend, int(dev_id)))
    
    # Convert to munet.Device objects
    device_type_map = {
        "Cuda": munet.DeviceType.CUDA,
        "Vulkan": munet.DeviceType.VULKAN,
        "Cpu": munet.DeviceType.CPU,
    }
    resolved = []
    for backend, dev_id in devices:
        dev_type = device_type_map[backend]
        candidate = munet.Device(dev_type, int(dev_id))
        try:
            # Probe allocation so invalid backend indices fail early and clearly.
            _ = munet.ones([1], device=candidate)
            resolved.append(candidate)
            continue
        except RuntimeError as err:
            if int(dev_id) != 0:
                fallback = munet.Device(dev_type, 0)
                try:
                    _ = munet.ones([1], device=fallback)
                    print(
                        f"[WARN] Requested {backend.lower()}:{dev_id} unavailable ({err}); "
                        f"falling back to {backend.lower()}:0"
                    )
                    resolved.append(fallback)
                    continue
                except RuntimeError:
                    pass
            raise
    return resolved


def make_one_hot(indices: np.ndarray, depth: int) -> np.ndarray:
    """Convert token indices to one-hot encoded tensor for Embedding layer.
    
    Args:
        indices: Token indices of shape [B, T] or [T]
        depth: Vocabulary size (number of classes)
    
    Returns:
        One-hot tensor of shape [B, T, depth] or [T, depth]
    """
    if indices.ndim == 1:
        indices = indices.reshape(1, -1)
        return np.eye(depth, dtype=np.float32)[indices].squeeze(0)
    return np.eye(depth, dtype=np.float32)[indices]


class TransformerBlock(nn.Module):
    """Single transformer decoder block."""

    def __init__(self, d_model: int, n_heads: int, device: munet.Device):
        super().__init__()
        self._device = device
        
        self.ln1 = nn.LayerNorm(d_model).to(device)
        self.attn = nn.MultiHeadAttention(d_model, n_heads, causal=True).to(device)
        self.ln2 = nn.LayerNorm(d_model).to(device)
        
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        ).to(device)

    def forward(self, x: munet.Tensor) -> munet.Tensor:
        # Pre-norm attention
        h = self.ln1(x)
        x = x + self.attn(h)
        # Pre-norm feed-forward
        h = self.ln2(x)
        x = x + self.ff(h)
        return x


class SimpleGPT(nn.Module):
    """Simple GPT model with layers distributed across multiple devices."""

    def __init__(self, vocab: int, ctx_len: int, d_model: int, n_heads: int, n_layers: int, devices):
        super().__init__()
        self.vocab = vocab
        self.ctx_len = ctx_len
        self.d_model = d_model
        self.devices = list(devices)
        
        # Embeddings on first device
        first_device = self.devices[0]
        self.token_embed = nn.Embedding(vocab, d_model).to(first_device)
        self.pos_embed = nn.Embedding(ctx_len, d_model).to(first_device)
        
        # Transformer blocks distributed across devices
        self.blocks = []
        for i in range(n_layers):
            device = self.devices[i % len(self.devices)]
            block = TransformerBlock(d_model, n_heads, device)
            self.blocks.append(block)
        
        # Final layer norm and output projection on last device
        last_device = self.devices[-1]
        self.ln_f = nn.LayerNorm(d_model).to(last_device)
        self.head = nn.Linear(d_model, vocab).to(last_device)

    def forward(self, x: munet.Tensor) -> munet.Tensor:
        """Forward pass through the model.
        
        Args:
            x: Combined input tensor [B, T, vocab + ctx_len] (token one-hot concatenated with position one-hot)
        
        Returns:
            Logits [B, T, vocab]
        """
        # Split combined input into token and position one-hot encodings
        # x shape: [batch, seq_len, vocab + ctx_len]
        tok_oh = x.narrow(2, 0, self.vocab)
        pos_oh = x.narrow(2, self.vocab, self.ctx_len)
        
        # Token + positional embeddings
        h = self.token_embed(tok_oh) + self.pos_embed(pos_oh)
        x = h
        
        # Transformer blocks with cross-device transfers
        for i, block in enumerate(self.blocks):
            x = block(x)
            # Move to next device if needed
            if i < len(self.blocks) - 1:
                next_device = self.blocks[i + 1]._device
                x = x.to(next_device)
        
        # Final projection to vocab
        x = self.ln_f(x)
        logits = self.head(x)
        return logits


def train(args):
    """Training loop."""
    # Parse devices
    devices = parse_device_spec(args.devices)
    print(f"Using devices: {devices}")
    
    # Sample corpus
    text = "hello world this is a simple language model training demo. " * 20
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for c, i in stoi.items()}
    vocab = len(chars)
    ctx = args.ctx_len
    
    # Build dataset
    ids = np.array([stoi[c] for c in text], dtype=np.int32)
    xs, ys = [], []
    for i in range(len(ids) - ctx):
        xs.append(ids[i : i + ctx])
        ys.append(ids[i + 1 : i + ctx + 1])
    X = np.array(xs, dtype=np.int32)
    Y = np.array(ys, dtype=np.int32)
    
    # Convert to one-hot for training (Embedding requires one-hot when requires_grad=True)
    X_oh = make_one_hot(X, vocab)
    Y_oh = make_one_hot(Y, vocab)
    # Position encoding one-hot
    P_oh = make_one_hot(np.arange(ctx)[None, :].repeat(X.shape[0], axis=0), ctx)
    
    model = SimpleGPT(
        vocab=vocab, ctx_len=ctx, d_model=args.d_model,
        n_heads=args.n_heads, n_layers=args.n_layers,
        devices=devices
    )
    opt = munet.optim.Adam(model.parameters(), lr=args.lr)
    
    print(f"Corpus: {len(ids)} tokens, vocab={vocab}")
    print(f"Model: d_model={args.d_model}, n_heads={args.n_heads}, n_layers={args.n_layers}")
    
    losses = []
    for step in range(args.steps):
        # Sample batch
        idx = np.random.randint(0, X.shape[0], size=args.batch_size)
        xb_oh = X_oh[idx]
        pb_oh = P_oh[idx]
        yb_oh = Y_oh[idx]
        
        # Combine token and position one-hot into single input tensor
        xb_combined = np.concatenate([xb_oh, pb_oh], axis=-1)  # [B, T, vocab + ctx_len]
        
        # Move to devices
        xb_tensor = munet.from_numpy(xb_combined).to(devices[0])
        # Target for cross-entropy
        yb_tensor = munet.from_numpy(yb_oh.reshape(-1, vocab)).to(devices[-1])
        
        # Forward pass
        logits = model(xb_tensor)
        logits_flat = logits.reshape([args.batch_size * ctx, vocab])
        
        # Loss and backward
        loss = logits_flat.cross_entropy(yb_tensor)
        opt.zero_grad()
        loss.backward()
        opt.step()
        
        losses.append(loss.item())
        if step % 10 == 0:
            avg_loss = sum(losses[-10:]) / min(10, len(losses))
            print(f"Step {step:4d} | Loss: {loss.item():.4f} | Avg(10): {avg_loss:.4f}")
    
    # Save model
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        model_path = os.path.join(args.output_dir, "model.bin")
        munet.save_checkpoint(model, model_path)
        print(f"Saved model to {model_path}")


def generate(args):
    """Generate text from prompt."""
    devices = parse_device_spec(args.devices)
    print(f"Using devices: {devices}")
    
    # Load model
    # TODO: implement model loading
    print("Model loading not yet implemented")


def main():
    parser = argparse.ArgumentParser(description="Multi-GPU GPT Demo")
    parser.add_argument("--devices", type=str, default="0:cuda",
                        help='Device specification, e.g., "0:cuda,1:vulkan"')
    
    subparsers = parser.add_subparsers(dest="command")
    
    # Train
    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--output-dir", type=str, default="./model")
    train_parser.add_argument("--steps", type=int, default=100)
    train_parser.add_argument("--batch-size", type=int, default=16)
    train_parser.add_argument("--lr", type=float, default=3e-2)
    train_parser.add_argument("--d-model", type=int, default=64)
    train_parser.add_argument("--n-heads", type=int, default=4)
    train_parser.add_argument("--n-layers", type=int, default=2)
    train_parser.add_argument("--ctx-len", type=int, default=16)
    train_parser.set_defaults(func=train)
    
    # Generate
    gen_parser = subparsers.add_parser("generate")
    gen_parser.add_argument("--model-dir", type=str, default="./model")
    gen_parser.add_argument("--prompt", type=str, default="hello")
    gen_parser.add_argument("--ctx-len", type=int, default=16)
    gen_parser.set_defaults(func=generate)
    
    args = parser.parse_args()
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
