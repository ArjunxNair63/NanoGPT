# NanoGPT Character Language Model

A small, from-scratch, decoder-only Transformer trained at the character level
on Tiny Shakespeare. The complete model and training loop are in `train.py`.

## Requirements

- Python 3.9 or newer
- PyTorch 2.0 or newer
- CUDA-enabled PyTorch and a compatible NVIDIA GPU for GPU acceleration

Install PyTorch using the command recommended for your operating system and
CUDA version at [pytorch.org](https://pytorch.org/get-started/locally/). Then
install the remaining project requirements:

```powershell
python -m pip install -r requirements.txt
```

## Train

```powershell
python train.py
```

The script downloads Tiny Shakespeare from Karpathy's `char-rnn` repository to
`input.txt` on the first run. It makes a 90/10 train/validation split, prints
losses every 500 steps, and trains for 5,000 steps by default. It selects CUDA
automatically when available, otherwise CPU.

At the end of training it prints a 1,000-character generated sample and saves:

- `checkpoint.pt` — model weights, vocabulary, and model configuration
- `generated_sample.txt` — the generated text

These output files and the downloaded corpus are local and excluded from Git.
The checkpoint is written by the current version of `train.py`; a training run
that completed before checkpoint saving was added cannot be recovered from its
printed output alone.

## Model

The model uses character token and learned position embeddings, six
Pre-LayerNorm Transformer blocks, six-head causal self-attention, GELU
feed-forward layers, residual connections, dropout, and AdamW. Default settings
are `block_size=256`, `batch_size=64`, `n_embd=384`, `n_head=6`, `n_layer=6`,
`dropout=0.2`, and `learning_rate=3e-4`.

## Dataset

Tiny Shakespeare is downloaded from:
<https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt>
