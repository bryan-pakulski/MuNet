"""Autoregressively sample characters from a trained TinyGPT checkpoint."""
import argparse
from pathlib import Path
import numpy as np
import munet as mu
from examples.common import positive_int, read_checkpoint
from .data import encode
from .model import TinyGPT


def sample(model, vocabulary, prompt, *, tokens=100, temperature=.8, top_k=20, seed=17, device="vulkan"):
    if not prompt or tokens < 0 or not np.isfinite(temperature) or temperature <= 0 or top_k < 0:
        raise ValueError("use a nonempty prompt, nonnegative token count/top-k and positive finite temperature")
    generated = encode(prompt, vocabulary).astype(int).tolist()
    rng = np.random.default_rng(seed)
    predict = mu.compile(model.eval(), device=device)
    characters = []
    for _ in range(tokens):
        window = generated[-model.context:]
        inputs = np.zeros((1, model.context), np.float32)
        # Right-padding leaves the final real prompt position unaffected by padding:
        # the causal mask prevents attention to all future positions.
        inputs[0, :len(window)] = window
        logits = predict(inputs).numpy()[0, len(window) - 1].astype(np.float64) / temperature
        if top_k and top_k < len(logits):
            keep = np.argsort(logits)[-top_k:]
            masked = np.full_like(logits, -np.inf)
            masked[keep] = logits[keep]
            logits = masked
        probabilities = np.exp(logits - logits.max())
        probabilities /= probabilities.sum()
        token = int(rng.choice(len(vocabulary), p=probabilities))
        generated.append(token)
        characters.append(vocabulary[token] if token else "\ufffd")
    return prompt + "".join(characters)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt", default="First Citizen:\n")
    parser.add_argument("--tokens", type=positive_int, default=200)
    parser.add_argument("--temperature", type=float, default=.8)
    parser.add_argument("--top-k", type=int, default=20, help="0 samples from the full vocabulary")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="vulkan")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    state = read_checkpoint(args.checkpoint, "language_model")
    model = TinyGPT(**state["config"]["model"])
    model.load_state_dict(state["model"])
    text = sample(model, state["config"]["vocabulary"], args.prompt, tokens=args.tokens,
                  temperature=args.temperature, top_k=args.top_k, seed=args.seed, device=args.device)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
