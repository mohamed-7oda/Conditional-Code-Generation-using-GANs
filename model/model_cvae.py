"""
model_cvae.py  –  Model 3 (outside course): Conditional Variational Autoencoder (CVAE).

Why CVAE?
    - Learns a *distribution* over valid dates for each condition set, not
      just a single mapping. At inference we sample from this distribution,
      giving diverse but valid outputs — exactly what a generative model should do.
    - The condition is concatenated to both encoder input and decoder input,
      so generation is explicitly conditioned on all 4 tokens.

Architecture:
    Encoder (recognition network) : [condition_emb ; date_emb] → μ, log σ²
    Decoder (generation network)  : [condition_emb ; z] → date tokens (autoregressive LSTM)
    Loss : reconstruction (cross-entropy) + KL divergence (annealed β)
"""

import random
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import build_vocab, PAD_IDX, SOS_IDX, EOS_IDX

char2idx, idx2char = build_vocab()
VOCAB_SIZE     = len(char2idx)
NUM_CONDITIONS = 4

# ── hyper-parameters ──────────────────────────────────────────────────────────
EMBED_SIZE   = 64
HIDDEN_SIZE  = 128
LATENT_SIZE  = 32
NUM_LAYERS   = 1
DROPOUT      = 0.2
BATCH_SIZE   = 512
EPOCHS       = 40
LR           = 1e-3
WEIGHTS_PATH = Path(__file__).parent / "cvae.pth"


class ConditionEmbedder(nn.Module):
    """Embeds 4 condition tokens → single fixed-size vector."""
    def __init__(self, vocab_size, embed_size, hidden_size, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=PAD_IDX)
        self.proj      = nn.Sequential(
            nn.Linear(embed_size * NUM_CONDITIONS, hidden_size),
            nn.Tanh(),
            nn.Dropout(dropout),
        )

    def forward(self, x):          # x: (B, 4)
        emb = self.embedding(x)    # (B, 4, E)
        return self.proj(emb.reshape(emb.size(0), -1))   # (B, H)


class CVAEEncoder(nn.Module):
    """Recognition network: q(z | x, c)"""
    def __init__(self, vocab_size, embed_size, hidden_size, latent_size, num_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=PAD_IDX)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        # condition + LSTM hidden → μ, log σ²
        self.fc_mu  = nn.Linear(hidden_size + hidden_size, latent_size)
        self.fc_var = nn.Linear(hidden_size + hidden_size, latent_size)

    def forward(self, date_ids, cond_vec):
        # date_ids: (B, T)  cond_vec: (B, H)
        emb = self.embedding(date_ids)
        _, (h, _) = self.lstm(emb)
        h = h[-1]                                    # (B, H) top layer
        combined = torch.cat([h, cond_vec], dim=1)   # (B, 2H)
        return self.fc_mu(combined), self.fc_var(combined)


class CVAEDecoder(nn.Module):
    """Generation network: p(x | z, c)"""
    def __init__(self, vocab_size, embed_size, hidden_size, latent_size, num_layers, dropout):
        super().__init__()
        self.embedding  = nn.Embedding(vocab_size, embed_size, padding_idx=PAD_IDX)
        self.z_proj     = nn.Linear(latent_size + hidden_size, hidden_size)  # [z;cond] → h0
        self.lstm       = nn.LSTM(embed_size + latent_size + hidden_size,   # input = [emb;z;cond]
                                  hidden_size, num_layers,
                                  batch_first=True,
                                  dropout=dropout if num_layers > 1 else 0.0)
        self.fc         = nn.Linear(hidden_size, vocab_size)
        self.dropout    = nn.Dropout(dropout)
        self.num_layers = num_layers
        self.hidden_size = hidden_size

    def forward_step(self, token, h, c, z_cond):
        # token: (B,)   z_cond: (B, latent+H)
        emb = self.dropout(self.embedding(token))           # (B, E)
        lstm_in = torch.cat([emb, z_cond], dim=1).unsqueeze(1)  # (B,1,E+lat+H)
        out, (h, c) = self.lstm(lstm_in, (h, c))
        logits = self.fc(out.squeeze(1))
        return logits, h, c

    def init_hidden(self, z, cond_vec):
        """Build initial hidden/cell state from z and condition."""
        z_cond = torch.cat([z, cond_vec], dim=1)
        h0 = torch.tanh(self.z_proj(z_cond))               # (B, H)
        # expand to (num_layers, B, H)
        h0 = h0.unsqueeze(0).expand(self.num_layers, -1, -1).contiguous()
        c0 = torch.zeros_like(h0)
        return h0, c0


class CVAE(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, latent_size, num_layers, dropout):
        super().__init__()
        self.cond_embedder = ConditionEmbedder(vocab_size, embed_size, hidden_size, dropout)
        self.encoder       = CVAEEncoder(vocab_size, embed_size, hidden_size, latent_size, num_layers, dropout)
        self.decoder       = CVAEDecoder(vocab_size, embed_size, hidden_size, latent_size, num_layers, dropout)
        self.vocab_size    = vocab_size
        self.latent_size   = latent_size

    @staticmethod
    def reparameterise(mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, src, trg, beta: float = 1.0):
        # src: (B,4)   trg: (B,T) with SOS..EOS
        cond_vec       = self.cond_embedder(src)              # (B,H)
        mu, log_var    = self.encoder(trg[:, 1:], cond_vec)   # encode date (no SOS)
        z              = self.reparameterise(mu, log_var)
        z_cond         = torch.cat([z, cond_vec], dim=1)

        B, T = trg.shape
        outputs = torch.zeros(B, T - 1, self.vocab_size, device=src.device)
        h, c    = self.decoder.init_hidden(z, cond_vec)
        token   = trg[:, 0]                                   # SOS

        for t in range(T - 1):
            logits, h, c = self.decoder.forward_step(token, h, c, z_cond)
            outputs[:, t] = logits
            token = trg[:, t + 1]                             # always teacher-force in VAE

        kl = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
        return outputs, kl

    @torch.no_grad()
    def generate_batch(self, src: torch.Tensor, max_len: int = 15, temperature: float = 1.0) -> list[list[int]]:
        self.eval()
        B        = src.size(0)
        cond_vec = self.cond_embedder(src)
        z        = torch.randn(B, self.latent_size, device=src.device)
        z_cond   = torch.cat([z, cond_vec], dim=1)
        h, c     = self.decoder.init_hidden(z, cond_vec)
        token    = torch.full((B,), SOS_IDX, dtype=torch.long, device=src.device)
        results  = [[] for _ in range(B)]
        done     = torch.zeros(B, dtype=torch.bool, device=src.device)

        for _ in range(max_len):
            logits, h, c = self.decoder.forward_step(token, h, c, z_cond)
            token = (logits / temperature).argmax(dim=-1)
            for i in range(B):
                if not done[i]:
                    if token[i].item() == EOS_IDX:
                        done[i] = True
                    else:
                        results[i].append(token[i].item())
            if done.all():
                break
        return results

    @torch.no_grad()
    def generate(self, src: torch.Tensor, max_len: int = 15) -> list[int]:
        return self.generate_batch(src, max_len)[0]


def build_model() -> CVAE:
    return CVAE(VOCAB_SIZE, EMBED_SIZE, HIDDEN_SIZE, LATENT_SIZE, NUM_LAYERS, DROPOUT)


def save_checkpoint(model: CVAE, path: Path = WEIGHTS_PATH) -> None:
    state = model._orig_mod.state_dict() if hasattr(model, "_orig_mod") else model.state_dict()
    torch.save({
        "model_state":  state,
        "vocab_size":   VOCAB_SIZE,
        "embed_size":   EMBED_SIZE,
        "hidden_size":  HIDDEN_SIZE,
        "latent_size":  LATENT_SIZE,
        "num_layers":   NUM_LAYERS,
        "dropout":      DROPOUT,
        "model_type":   "cvae",
    }, path)


def load_checkpoint(path: Path = WEIGHTS_PATH, device: torch.device = torch.device("cpu")) -> CVAE:
    ckpt  = torch.load(path, map_location=device)
    model = CVAE(
        ckpt["vocab_size"], ckpt["embed_size"], ckpt["hidden_size"],
        ckpt["latent_size"], ckpt["num_layers"], ckpt["dropout"],
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model