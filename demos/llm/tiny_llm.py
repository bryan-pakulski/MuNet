import os
import sys
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "../../build"))
import munet


class TinyLLM(munet.nn.Module):
    """
    Minimal char-level LM demo that is transformer-adjacent:
    - token one-hot -> nn.Embedding
    - flatten context -> MLP head
    - cross entropy loss on one-hot target

    Note: this is a pragmatic stepping stone toward a full Transformer stack.
    """

    def __init__(self, vocab_size: int, context_len: int, d_model: int = 32, hidden: int = 64):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_len = context_len
        self.embed = munet.nn.Embedding(vocab_size, d_model)
        self.pos_embed = munet.nn.Embedding(context_len, d_model)
        self.ln = munet.nn.LayerNorm(d_model)
        self.gelu = munet.nn.GELU()
        self.fc1 = munet.nn.Linear(context_len * d_model, hidden)
        self.fc2 = munet.nn.Linear(hidden, vocab_size)

    def forward(self, x_onehot, pos_onehot):
        # x_onehot: [B, T, V], pos_onehot: [B, T, T]
        tok = self.embed(x_onehot)
        pos = self.pos_embed(pos_onehot)
        x = tok + pos
        x = self.ln(x)
        x = x.reshape([x.shape[0], self.context_len * x.shape[2]])
        x = self.gelu(self.fc1(x))
        return self.fc2(x)


def one_hot(indices: np.ndarray, depth: int):
    out = np.zeros((indices.shape[0], depth), dtype=np.float32)
    out[np.arange(indices.shape[0]), indices] = 1.0
    return out


def make_dataset(text: str, context_len: int):
    chars = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}

    tokens = np.array([stoi[c] for c in text], dtype=np.int32)
    xs, ys = [], []
    for i in range(len(tokens) - context_len):
        xs.append(tokens[i : i + context_len])
        ys.append(tokens[i + context_len])

    X = np.array(xs, dtype=np.int32)
    y = np.array(ys, dtype=np.int32)
    return X, y, stoi, itos


def sample_next(model: TinyLLM, context_tokens: np.ndarray, vocab_size: int):
    x_oh = np.zeros((1, context_tokens.shape[0], vocab_size), dtype=np.float32)
    for t, idx in enumerate(context_tokens):
        x_oh[0, t, idx] = 1.0

    pos_oh = np.zeros((1, context_tokens.shape[0], context_tokens.shape[0]), dtype=np.float32)
    for t in range(context_tokens.shape[0]):
        pos_oh[0, t, t] = 1.0

    logits = model.forward(munet.from_numpy(x_oh), munet.from_numpy(pos_oh)).to(
        munet.Device(munet.DeviceType.CPU, 0)
    )
    probs = np.array(logits.softmax(), copy=False)[0]
    return int(np.argmax(probs))


def main():
    text = (
        "to be, or not to be, that is the question. "
        "whether tis nobler in the mind to suffer. "
    )
    context_len = 8
    X, y, stoi, itos = make_dataset(text, context_len)
    vocab = len(stoi)

    model = TinyLLM(vocab, context_len, d_model=24, hidden=64)
    opt = munet.optim.Adam(model.parameters(), lr=3e-3)

    # Pre-build one-hot dataset for speed.
    X_oh = np.zeros((X.shape[0], context_len, vocab), dtype=np.float32)
    for i in range(X.shape[0]):
        for t in range(context_len):
            X_oh[i, t, X[i, t]] = 1.0
    y_oh = one_hot(y, vocab)
    pos_oh = np.zeros((X.shape[0], context_len, context_len), dtype=np.float32)
    for i in range(X.shape[0]):
        for t in range(context_len):
            pos_oh[i, t, t] = 1.0

    bs = 16
    steps = 200
    for step in range(steps):
        idx = np.random.randint(0, X_oh.shape[0], size=bs)
        xb = munet.from_numpy(X_oh[idx])
        pb = munet.from_numpy(pos_oh[idx])
        yb = munet.from_numpy(y_oh[idx])

        opt.zero_grad()
        logits = model.forward(xb, pb)
        loss = logits.cross_entropy(yb)
        loss.backward()
        opt.step()

        if step % 40 == 0:
            print(f"step {step:4d} | loss {loss.item():.4f}")

    # Inference demo
    seed = "to be, o"
    ctx = np.array([stoi[c] for c in seed[-context_len:]], dtype=np.int32)
    generated = list(seed)

    for _ in range(80):
        nxt = sample_next(model, ctx, vocab)
        generated.append(itos[nxt])
        ctx = np.concatenate([ctx[1:], np.array([nxt], dtype=np.int32)])

    print("\n=== Generated ===")
    print("".join(generated))


if __name__ == "__main__":
    main()
