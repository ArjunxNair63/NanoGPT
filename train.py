"""Character-level decoder-only Transformer for Tiny Shakespeare.

Run with ``python train.py``. The dataset is downloaded on the first run and
cached beside this script as ``input.txt``.
"""

from pathlib import Path
import urllib.request

import torch
import torch.nn as nn
from torch.nn import functional as F


# Hyperparameters
batch_size = 64
block_size = 256
max_iters = 5000
eval_interval = 500
eval_iters = 200
learning_rate = 3e-4
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2

DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DATA_PATH = Path(__file__).with_name("input.txt")
CHECKPOINT_PATH = Path(__file__).with_name("checkpoint.pt")
SAMPLE_PATH = Path(__file__).with_name("generated_sample.txt")
device = "cuda" if torch.cuda.is_available() else "cpu"


def load_text():
    """Download the corpus once and return its UTF-8 text."""
    if not DATA_PATH.exists():
        print(f"Downloading Tiny Shakespeare to {DATA_PATH}...")
        urllib.request.urlretrieve(DATA_URL, DATA_PATH)
    return DATA_PATH.read_text(encoding="utf-8")


text = load_text()
chars = sorted(set(text))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}


def encode(s):
    """Convert a string into a list of character token IDs."""
    return [stoi[ch] for ch in s]


def decode(ids):
    """Convert character token IDs back into a string."""
    return "".join(itos[int(i)] for i in ids)


data = torch.tensor(encode(text), dtype=torch.long)
split_at = int(0.9 * len(data))
train_data = data[:split_at]
val_data = data[split_at:]


def get_batch(split):
    """Sample random input/next-character target windows from a split."""
    source = train_data if split == "train" else val_data
    if len(source) <= block_size:
        raise ValueError(f"The {split} split must contain more than block_size tokens")
    starts = torch.randint(len(source) - block_size, (batch_size,))
    x = torch.stack([source[i : i + block_size] for i in starts])
    y = torch.stack([source[i + 1 : i + block_size + 1] for i in starts])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model):
    """Estimate mean train and validation losses over fresh random batches."""
    model.eval()
    losses = {}
    for split in ("train", "val"):
        batch_losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(split)
            _, loss = model(x, y)
            batch_losses[k] = loss.item()
        losses[split] = batch_losses.mean().item()
    model.train()
    return losses


class Head(nn.Module):
    """One self-attention head with a causal mask."""

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.attn_dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch, time, channels = x.shape
        key = self.key(x)
        query = self.query(x)
        # Scaled dot-product attention: (B,T,C) @ (B,C,T) -> (B,T,T).
        weights = query @ key.transpose(-2, -1) * (key.shape[-1] ** -0.5)
        weights = weights.masked_fill(self.tril[:time, :time] == 0, float("-inf"))
        weights = F.softmax(weights, dim=-1)
        weights = self.attn_dropout(weights)
        value = self.value(x)
        return weights @ value


class MultiHeadAttention(nn.Module):
    """Run multiple self-attention heads in parallel and project their output."""

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(num_heads * head_size, n_embd)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x):
        joined = torch.cat([head(x) for head in self.heads], dim=-1)
        return self.resid_dropout(self.proj(joined))


class FeedForward(nn.Module):
    """Position-wise MLP used after attention in each Transformer block."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """Pre-LayerNorm Transformer block with residual connections."""

    def __init__(self):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class GPTLanguageModel(nn.Module):
    """Decoder-only character language model."""

    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block() for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        batch, time = idx.shape
        if time > block_size:
            raise ValueError(f"Sequence length {time} exceeds block_size={block_size}")
        positions = torch.arange(time, device=idx.device)
        x = self.token_embedding_table(idx) + self.position_embedding_table(positions)
        x = self.blocks(x)
        logits = self.lm_head(self.ln_f(x))

        loss = None
        if targets is not None:
            batch, time, channels = logits.shape
            loss = F.cross_entropy(logits.view(batch * time, channels), targets.view(batch * time))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        """Generate tokens autoregressively, cropping context to block_size."""
        for _ in range(max_new_tokens):
            context = idx[:, -block_size:]
            logits, _ = self(context)
            probabilities = F.softmax(logits[:, -1, :], dim=-1)
            next_token = torch.multinomial(probabilities, num_samples=1)
            idx = torch.cat((idx, next_token), dim=1)
        return idx


def main():
    print(f"Device: {device}; vocabulary: {vocab_size} characters; corpus: {len(data):,} tokens")
    model = GPTLanguageModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    for step in range(max_iters):
        if step % eval_interval == 0 or step == max_iters - 1:
            losses = estimate_loss(model)
            print(f"step {step:5d}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        xb, yb = get_batch("train")
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    model.eval()
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    sample = decode(model.generate(context, max_new_tokens=1000)[0].tolist())
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "chars": chars,
            "config": {
                "batch_size": batch_size,
                "block_size": block_size,
                "n_embd": n_embd,
                "n_head": n_head,
                "n_layer": n_layer,
                "dropout": dropout,
                "learning_rate": learning_rate,
                "max_iters": max_iters,
            },
        },
        CHECKPOINT_PATH,
    )
    SAMPLE_PATH.write_text(sample, encoding="utf-8")
    print(f"Saved model checkpoint to {CHECKPOINT_PATH}")
    print(f"Saved generated sample to {SAMPLE_PATH}")
    print("\n--- Generated sample ---")
    print(sample)


if __name__ == "__main__":
    main()
