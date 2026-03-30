import numpy as np

import munet


class DecoderBlock(munet.nn.Module):
    def __init__(self, d_model: int, n_heads: int, ff_hidden: int):
        super().__init__()
        self.ln1 = munet.nn.LayerNorm(d_model)
        self.attn = munet.nn.MultiHeadAttention(d_model, n_heads, causal=True)
        self.ln2 = munet.nn.LayerNorm(d_model)
        self.fc1 = munet.nn.Linear(d_model, ff_hidden)
        self.gelu = munet.nn.GELU()
        self.fc2 = munet.nn.Linear(ff_hidden, d_model)

    def forward(self, x):
        h = self.ln1(x)
        x = x + self.attn(h)
        h2 = self.ln2(x)
        ff = self.fc2(
            self.gelu(self.fc1(h2.reshape([h2.shape[0] * h2.shape[1], h2.shape[2]])))
        )
        ff = ff.reshape([h2.shape[0], h2.shape[1], h2.shape[2]])
        return x + ff


class TinyDecoderLM(munet.nn.Module):
    def __init__(self, vocab: int, ctx: int, d_model: int = 32):
        super().__init__()
        self.vocab = vocab
        self.ctx = ctx
        self.token = munet.nn.Embedding(vocab, d_model)
        self.pos = munet.nn.Embedding(ctx, d_model)
        self.block = DecoderBlock(d_model=d_model, n_heads=4, ff_hidden=64)
        self.ln_f = munet.nn.LayerNorm(d_model)
        self.head = munet.nn.Linear(d_model, vocab)

    def forward(self, tok_oh, pos_oh):
        x = self.token(tok_oh) + self.pos(pos_oh)
        x = self.block(x)
        x = self.ln_f(x)
        # Next-token head on final position only
        last = x.reshape([x.shape[0] * x.shape[1], x.shape[2]])
        logits = self.head(last).reshape([x.shape[0], x.shape[1], self.vocab])
        return logits


def make_one_hot(tokens: np.ndarray, vocab: int):
    out = np.zeros((tokens.shape[0], tokens.shape[1], vocab), dtype=np.float32)
    for b in range(tokens.shape[0]):
        for t in range(tokens.shape[1]):
            out[b, t, tokens[b, t]] = 1.0
    return out


def make_pos(batch: int, ctx: int):
    out = np.zeros((batch, ctx, ctx), dtype=np.float32)
    for b in range(batch):
        for t in range(ctx):
            out[b, t, t] = 1.0
    return out


def main():
    text = "transformers are getting closer in munet. "
    vocab_chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(vocab_chars)}
    itos = {i: c for c, i in stoi.items()}
    ids = np.array([stoi[c] for c in text], dtype=np.int32)

    ctx = 8
    xs, ys = [], []
    for i in range(len(ids) - ctx):
        xs.append(ids[i : i + ctx])
        ys.append(ids[i + 1 : i + ctx + 1])
    X = np.array(xs, dtype=np.int32)
    Y = np.array(ys, dtype=np.int32)

    vocab = len(vocab_chars)
    model = TinyDecoderLM(vocab, ctx, d_model=32)
    opt = munet.optim.Adam(model.parameters(), lr=3e-3)

    X_oh = make_one_hot(X, vocab)
    Y_oh = make_one_hot(Y, vocab)
    P_oh = make_pos(X.shape[0], ctx)

    for step in range(120):
        idx = np.random.randint(0, X.shape[0], size=16)
        xb = munet.from_numpy(X_oh[idx])
        pb = munet.from_numpy(P_oh[idx])
        yb = munet.from_numpy(Y_oh[idx]).reshape([16 * ctx, vocab])

        opt.zero_grad()
        logits = model.forward(xb, pb).reshape([16 * ctx, vocab])
        loss = logits.cross_entropy(yb)
        loss.backward()
        opt.step()

        if step % 30 == 0:
            print(f"step {step:3d} | loss={loss.item():.4f}")

    # simple generation
    seed = "transfor"
    ctx_ids = np.array([stoi[c] for c in seed[-ctx:]], dtype=np.int32)
    out = list(seed)
    for _ in range(60):
        tok = make_one_hot(ctx_ids.reshape(1, -1), vocab)
        pos = make_pos(1, ctx)
        with munet.no_grad():
            logits = model.forward(munet.from_numpy(tok), munet.from_numpy(pos))
            probs = np.array(logits.softmax(-1).detach(), copy=False)[0, -1]
        nxt = int(np.argmax(probs))
        out.append(itos[nxt])
        ctx_ids = np.concatenate([ctx_ids[1:], np.array([nxt], dtype=np.int32)])

    print("\nGenerated:\n" + "".join(out))


if __name__ == "__main__":
    main()
