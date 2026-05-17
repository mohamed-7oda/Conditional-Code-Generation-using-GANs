"""
model_cgan.py  –  Model 4 (course, required): Conditional GAN (cGAN).

Why cGAN?
    - The assignment explicitly requires a GAN.
    - Generator takes [condition_embedding ; noise] → date token sequence.
    - Discriminator takes [condition_embedding ; date_embedding] → real/fake.
    - Training alternates G and D updates (standard GAN training loop).

Architecture:
    Generator   : MLP condition encoder + LSTM decoder conditioned on noise+cond
    Discriminator : MLP that reads condition + LSTM-encoded date → sigmoid score
"""

import random
from pathlib import Path

import torch
import torch.nn as nn

from utils import build_vocab, PAD_IDX, SOS_IDX, EOS_IDX

char2idx, idx2char = build_vocab()
VOCAB_SIZE     = len(char2idx)
NUM_CONDITIONS = 4

# ── hyper-parameters ──────────────────────────────────────────────────────────
EMBED_SIZE   = 64
HIDDEN_SIZE  = 128
NOISE_SIZE   = 32
NUM_LAYERS   = 1
DROPOUT      = 0.2
BATCH_SIZE   = 512
EPOCHS       = 40
LR_G         = 2e-4
LR_D         = 1e-4
WEIGHTS_PATH = Path(__file__).parent / "cgan.pth"


# ── shared condition embedder ─────────────────────────────────────────────────

class ConditionEmbedder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=PAD_IDX)
        self.proj = nn.Sequential(
            nn.Linear(embed_size * NUM_CONDITIONS, hidden_size),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        emb = self.embedding(x)
        return self.proj(emb.reshape(emb.size(0), -1))   # (B, H)


# ── generator ─────────────────────────────────────────────────────────────────

class Generator(nn.Module):
    """
    G(z, c) → date token sequence (hard argmax, straight-through for training).
    At inference: greedy decode.
    During training: we generate integer sequences and feed real/fake to D.
    """
    def __init__(self, vocab_size, embed_size, hidden_size, noise_size, num_layers, dropout):
        super().__init__()
        self.cond_embed  = ConditionEmbedder(vocab_size, embed_size, hidden_size, dropout)
        self.noise_proj  = nn.Linear(noise_size, hidden_size)
        # initial hidden state from noise+cond
        self.to_hidden   = nn.Linear(hidden_size * 2, hidden_size * num_layers)
        self.to_cell     = nn.Linear(hidden_size * 2, hidden_size * num_layers)
        self.token_embed = nn.Embedding(vocab_size, embed_size, padding_idx=PAD_IDX)
        self.lstm        = nn.LSTM(embed_size + hidden_size,   # [token_emb ; cond]
                                   hidden_size, num_layers,
                                   batch_first=True,
                                   dropout=dropout if num_layers > 1 else 0.0)
        self.fc          = nn.Linear(hidden_size, vocab_size)
        self.dropout     = nn.Dropout(dropout)
        self.noise_size  = noise_size
        self.num_layers  = num_layers
        self.hidden_size = hidden_size
        self.vocab_size  = vocab_size

    def forward(self, src, max_len: int = 15, temperature: float = 1.0):
        """Generate token id sequences — greedy decode."""
        B        = src.size(0)
        cond_vec = self.cond_embed(src)                              # (B, H)
        z        = torch.randn(B, self.noise_size, device=src.device)
        nz       = torch.tanh(self.noise_proj(z))                   # (B, H)
        combined = torch.cat([cond_vec, nz], dim=1)                 # (B, 2H)

        L = self.num_layers
        h = torch.tanh(self.to_hidden(combined)).view(B, L, self.hidden_size).permute(1,0,2).contiguous()
        c = torch.tanh(self.to_cell(combined)).view(B, L, self.hidden_size).permute(1,0,2).contiguous()

        token   = torch.full((B,), SOS_IDX, dtype=torch.long, device=src.device)
        results = [[] for _ in range(B)]
        done    = torch.zeros(B, dtype=torch.bool, device=src.device)
        all_ids = []

        for _ in range(max_len):
            emb     = self.dropout(self.token_embed(token))
            lstm_in = torch.cat([emb, cond_vec], dim=1).unsqueeze(1)
            out, (h, c) = self.lstm(lstm_in, (h, c))
            logits  = self.fc(out.squeeze(1)) / temperature
            token   = logits.argmax(dim=-1)
            all_ids.append(token)

            for i in range(B):
                if not done[i]:
                    if token[i].item() == EOS_IDX:
                        done[i] = True
                    else:
                        results[i].append(token[i].item())
            if done.all():
                break

        # pad all_ids to same length and stack → (B, T)
        max_t    = len(all_ids)
        id_tensor = torch.stack(all_ids, dim=1)   # (B, T)
        return id_tensor, results

    def generate_batch(self, src, max_len=15):
        _, results = self.forward(src, max_len)
        return results

    def generate(self, src, max_len=15):
        return self.generate_batch(src, max_len)[0]


# ── discriminator ─────────────────────────────────────────────────────────────

class Discriminator(nn.Module):
    """D(date_ids, c) → probability of being real."""
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers, dropout):
        super().__init__()
        self.cond_embed = ConditionEmbedder(vocab_size, embed_size, hidden_size, dropout)
        self.date_embed = nn.Embedding(vocab_size, embed_size, padding_idx=PAD_IDX)
        self.lstm       = nn.LSTM(embed_size, hidden_size, num_layers,
                                  batch_first=True,
                                  dropout=dropout if num_layers > 1 else 0.0)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, src, date_ids):
        # src: (B,4)   date_ids: (B,T) token ids
        cond_vec = self.cond_embed(src)
        emb      = self.date_embed(date_ids)
        _, (h, _) = self.lstm(emb)
        h        = h[-1]                                     # (B, H)
        combined = torch.cat([h, cond_vec], dim=1)
        return self.fc(combined).squeeze(1)                  # (B,) logits


class ConditionalGAN(nn.Module):
    """Wrapper holding G and D for easy checkpoint save/load."""
    def __init__(self, vocab_size, embed_size, hidden_size, noise_size, num_layers, dropout):
        super().__init__()
        self.generator     = Generator(vocab_size, embed_size, hidden_size, noise_size, num_layers, dropout)
        self.discriminator = Discriminator(vocab_size, embed_size, hidden_size, num_layers, dropout)
        self.vocab_size    = vocab_size

    # Delegate generate helpers to G
    def generate_batch(self, src, max_len=15):
        return self.generator.generate_batch(src, max_len)

    def generate(self, src, max_len=15):
        return self.generator.generate(src, max_len)


def build_model() -> ConditionalGAN:
    return ConditionalGAN(VOCAB_SIZE, EMBED_SIZE, HIDDEN_SIZE, NOISE_SIZE, NUM_LAYERS, DROPOUT)


def save_checkpoint(model: ConditionalGAN, path: Path = WEIGHTS_PATH) -> None:
    g_state = model.generator._orig_mod.state_dict() if hasattr(model.generator, "_orig_mod") else model.generator.state_dict()
    d_state = model.discriminator._orig_mod.state_dict() if hasattr(model.discriminator, "_orig_mod") else model.discriminator.state_dict()
    torch.save({
        "g_state":     g_state,
        "d_state":     d_state,
        "vocab_size":  VOCAB_SIZE,
        "embed_size":  EMBED_SIZE,
        "hidden_size": HIDDEN_SIZE,
        "noise_size":  NOISE_SIZE,
        "num_layers":  NUM_LAYERS,
        "dropout":     DROPOUT,
        "model_type":  "cgan",
    }, path)


def load_checkpoint(path: Path = WEIGHTS_PATH, device: torch.device = torch.device("cpu")) -> ConditionalGAN:
    ckpt  = torch.load(path, map_location=device)
    model = ConditionalGAN(
        ckpt["vocab_size"], ckpt["embed_size"], ckpt["hidden_size"],
        ckpt["noise_size"], ckpt["num_layers"], ckpt["dropout"],
    ).to(device)
    model.generator.load_state_dict(ckpt["g_state"])
    model.discriminator.load_state_dict(ckpt["d_state"])
    model.eval()
    return model