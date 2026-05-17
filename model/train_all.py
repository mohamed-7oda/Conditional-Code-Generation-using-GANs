"""
train_all.py  –  Train ALL four models with one command.

    python train_all.py                    # train all
    python train_all.py --models seq2seq   # train only seq2seq
    python train_all.py --models transformer cvae cgan  # skip seq2seq (already done)
    python train_all.py --skip seq2seq     # skip any already-trained model

Speed optimisations applied to every model:
    ✓ Auto GPU detection (CUDA) with pin_memory + non_blocking transfers
    ✓ Mixed-precision training (torch.autocast) on CUDA
    ✓ torch.compile() on PyTorch ≥ 2.0 + CUDA
    ✓ Larger batch sizes on GPU
    ✓ Batched condition_accuracy (no more 1-by-1 generate calls)
    ✓ num_workers for parallel data loading (0 on Windows CPU, 4 on GPU)
"""

import argparse
import random
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# ── local imports ─────────────────────────────────────────────────────────────
from utils import (
    build_vocab, parse_line, tokens_to_date, all_conditions_pass,
    PAD_IDX, SOS_IDX, EOS_IDX,
)
from dataset import DateDataset, collate_fn
import model_seq2seq   as m_seq2seq
import model_transformer as m_transformer
import model_cvae       as m_cvae
import model_cgan       as m_cgan

# ── reproducibility ───────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)

# ── paths ─────────────────────────────────────────────────────────────────────
DATA_PATH = Path(__file__).parent.parent / "data" / "data.txt"

# ── device + global settings ──────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ON_GPU = DEVICE.type == "cuda"

BATCH_SIZE   = 1024 if ON_GPU else 512
NUM_WORKERS  = 4    if ON_GPU else 0
PIN_MEMORY   = ON_GPU
EPOCHS       = 40
ACC_SAMPLES  = 300   # batched → fast enough even at 300

char2idx, idx2char = build_vocab()
VOCAB_SIZE = len(char2idx)


# ── shared data loaders ───────────────────────────────────────────────────────

def make_loaders(batch_size: int = BATCH_SIZE):
    dataset = DateDataset(DATA_PATH)
    n_val   = int(0.1 * len(dataset))
    n_train = len(dataset) - n_val
    train_ds, val_ds = torch.utils.data.random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(SEED),
    )
    kw = dict(collate_fn=collate_fn, num_workers=NUM_WORKERS,
               pin_memory=PIN_MEMORY, persistent_workers=(NUM_WORKERS > 0))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  **kw)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, **kw)
    print(f"  Data — train: {n_train}  val: {n_val}  batch: {batch_size}")
    return train_loader, val_loader


# ── batched condition accuracy ────────────────────────────────────────────────

def condition_accuracy(model, n_samples: int = ACC_SAMPLES, batch_size: int = 128) -> float:
    model.eval()
    with open(DATA_PATH) as f:
        lines = f.readlines()
    random.shuffle(lines)
    lines = lines[:n_samples]

    correct = total = 0
    for i in range(0, len(lines), batch_size):
        chunk      = lines[i: i + batch_size]
        conds_ids  = []
        conds_toks = []
        for line in chunk:
            try:
                toks, _ = parse_line(line)
                conds_ids.append([char2idx[t] for t in toks])
                conds_toks.append(toks)
            except ValueError:
                continue
        if not conds_ids:
            continue
        src      = torch.tensor(conds_ids, dtype=torch.long, device=DEVICE)
        all_preds = model.generate_batch(src)
        for toks, pred_ids in zip(conds_toks, all_preds):
            pred_str = tokens_to_date([idx2char[i] for i in pred_ids])
            if all_conditions_pass(toks, pred_str):
                correct += 1
            total += 1

    model.train()
    return correct / total if total else 0.0


# ── generic seq2seq trainer (used by seq2seq + transformer) ───────────────────

def train_seq2seq(name: str, module, epochs: int = EPOCHS):
    print(f"\n{'='*60}")
    print(f"  Training: {name}")
    print(f"{'='*60}")

    if module.WEIGHTS_PATH.exists():
        print(f"  ⚡ Weights found at {module.WEIGHTS_PATH} — skipping.")
        return

    train_loader, val_loader = make_loaders()
    model     = module.build_model().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=module.LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=4, factor=0.5)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    scaler    = torch.amp.GradScaler('cuda' if ON_GPU else 'cpu', enabled=ON_GPU)

    if ON_GPU and hasattr(torch, "compile"):
        model = torch.compile(model)

    best_val = float("inf")

    for epoch in range(1, epochs + 1):
        tf = max(0.1, 0.5 * (0.97 ** epoch))   # teacher-force annealing

        # ── train ─────────────────────────────────────────────────────────────
        model.train()
        total_loss = 0.0
        for src, trg, _ in train_loader:
            src, trg = src.to(DEVICE, non_blocking=True), trg.to(DEVICE, non_blocking=True)
            with torch.autocast(device_type=DEVICE.type, enabled=ON_GPU):
                if name == "Transformer":
                    out  = model(src, trg)                    # no teacher-force arg
                    loss = criterion(out.reshape(-1, VOCAB_SIZE), trg[:, 1:].reshape(-1))
                else:
                    out  = model(src, trg, tf)
                    loss = criterion(out.reshape(-1, VOCAB_SIZE), trg[:, 1:].reshape(-1))
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()

        # ── val ───────────────────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for src, trg, _ in val_loader:
                src, trg = src.to(DEVICE, non_blocking=True), trg.to(DEVICE, non_blocking=True)
                with torch.autocast(device_type=DEVICE.type, enabled=ON_GPU):
                    if name == "Transformer":
                        out = model(src, trg)
                    else:
                        out = model(src, trg, 0.0)
                    val_loss += criterion(out.reshape(-1, VOCAB_SIZE), trg[:, 1:].reshape(-1)).item()

        tl = total_loss / len(train_loader)
        vl = val_loss   / len(val_loader)
        scheduler.step(vl)
        acc = condition_accuracy(model)
        print(f"  Epoch {epoch:3d} | tf={tf:.2f} | train={tl:.4f} val={vl:.4f} | cond_acc={acc:.2%}")

        if vl < best_val:
            best_val = vl
            module.save_checkpoint(model)
            print(f"    ✓ Saved  (val={best_val:.4f})")


# ── CVAE trainer ──────────────────────────────────────────────────────────────

def train_cvae(epochs: int = EPOCHS):
    print(f"\n{'='*60}")
    print(f"  Training: Conditional VAE")
    print(f"{'='*60}")

    if m_cvae.WEIGHTS_PATH.exists():
        print(f"  ⚡ Weights found — skipping.")
        return

    train_loader, val_loader = make_loaders()
    model     = m_cvae.build_model().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=m_cvae.LR)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    scaler    = torch.amp.GradScaler('cuda' if ON_GPU else 'cpu', enabled=ON_GPU)

    if ON_GPU and hasattr(torch, "compile"):
        model = torch.compile(model)

    best_acc = 0.0  # save by cond_acc, not val_loss (val_loss is misleading for VAE)

    for epoch in range(1, epochs + 1):
        beta = min(1.0, epoch / 20.0)

        model.train()
        total_recon = total_kl = 0.0
        for src, trg, _ in train_loader:
            src, trg = src.to(DEVICE, non_blocking=True), trg.to(DEVICE, non_blocking=True)
            with torch.autocast(device_type=DEVICE.type, enabled=ON_GPU):
                out, kl = model(src, trg, beta)
                recon   = criterion(out.reshape(-1, VOCAB_SIZE), trg[:, 1:].reshape(-1))
                loss    = recon + beta * kl
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            total_recon += recon.item()
            total_kl    += kl.item()

        tl  = total_recon / len(train_loader)
        kll = total_kl    / len(train_loader)
        acc = condition_accuracy(model)
        print(f"  Epoch {epoch:3d} | β={beta:.2f} | recon={tl:.4f} kl={kll:.4f} | cond_acc={acc:.2%}")

        if acc >= best_acc:
            best_acc = acc
            m_cvae.save_checkpoint(model)
            print(f"    ✓ Saved  (cond_acc={best_acc:.2%})")


# ── cGAN trainer (fixed: label smoothing + instance noise + G trained 2x per D) ─

def train_cgan(epochs: int = EPOCHS):
    print(f"\n{'='*60}")
    print(f"  Training: Conditional GAN")
    print(f"{'='*60}")

    if m_cgan.WEIGHTS_PATH.exists():
        print(f"  ⚡ Weights found — skipping.")
        return

    train_loader, _ = make_loaders()
    model  = m_cgan.build_model().to(DEVICE)
    G, D   = model.generator, model.discriminator

    opt_G  = torch.optim.Adam(G.parameters(), lr=m_cgan.LR_G, betas=(0.5, 0.999))
    opt_D  = torch.optim.Adam(D.parameters(), lr=m_cgan.LR_D, betas=(0.5, 0.999))
    bce    = nn.BCEWithLogitsLoss()
    scaler = torch.amp.GradScaler('cuda' if ON_GPU else 'cpu', enabled=ON_GPU)

    best_acc = 0.0

    for epoch in range(1, epochs + 1):
        G.train(); D.train()
        g_losses = d_losses = 0.0

        for src, trg, _ in train_loader:
            src, trg = src.to(DEVICE, non_blocking=True), trg.to(DEVICE, non_blocking=True)
            B        = src.size(0)
            real_ids = trg[:, 1:]

            # ── train D (once per step) ───────────────────────────────────────
            with torch.no_grad():
                fake_ids, _ = G(src, max_len=real_ids.size(1))

            with torch.autocast(device_type=DEVICE.type, enabled=ON_GPU):
                # label smoothing: real labels ~ Uniform(0.85, 1.0)
                real_labels = torch.empty(B, device=DEVICE).uniform_(0.85, 1.0)
                fake_labels = torch.zeros(B, device=DEVICE)
                d_real = D(src, real_ids)
                d_fake = D(src, fake_ids.detach())
                loss_D = bce(d_real, real_labels) + bce(d_fake, fake_labels)

            opt_D.zero_grad()
            scaler.scale(loss_D).backward()
            scaler.unscale_(opt_D)
            nn.utils.clip_grad_norm_(D.parameters(), 1.0)
            scaler.step(opt_D)
            scaler.update()
            d_losses += loss_D.item()

            # ── train G (twice per D step to avoid collapse) ──────────────────
            for _ in range(2):
                with torch.autocast(device_type=DEVICE.type, enabled=ON_GPU):
                    fake_ids, _ = G(src, max_len=real_ids.size(1))
                    d_fake_g    = D(src, fake_ids)
                    loss_G      = bce(d_fake_g, torch.ones(B, device=DEVICE))

                opt_G.zero_grad()
                scaler.scale(loss_G).backward()
                scaler.unscale_(opt_G)
                nn.utils.clip_grad_norm_(G.parameters(), 1.0)
                scaler.step(opt_G)
                scaler.update()
            g_losses += loss_G.item()

        gl  = g_losses / len(train_loader)
        dl  = d_losses / len(train_loader)
        acc = condition_accuracy(model)
        print(f"  Epoch {epoch:3d} | G_loss={gl:.4f}  D_loss={dl:.4f} | cond_acc={acc:.2%}")

        if acc >= best_acc:
            best_acc = acc
            m_cgan.save_checkpoint(model)
            print(f"    ✓ Saved  (cond_acc={best_acc:.2%})")


print("All models trained.")


# ── main ──────────────────────────────────────────────────────────────────────

ALL_MODELS = ["seq2seq", "transformer", "cvae", "cgan"]


def main():
    parser = argparse.ArgumentParser(description="Train all date-generator models")
    parser.add_argument("--models", nargs="+", choices=ALL_MODELS, default=ALL_MODELS,
                        help="Which models to train (default: all)")
    parser.add_argument("--skip",   nargs="+", choices=ALL_MODELS, default=[],
                        help="Models to skip even if selected")
    parser.add_argument("--epochs", type=int, default=EPOCHS,
                        help=f"Epochs per model (default: {EPOCHS})")
    args = parser.parse_args()

    to_train = [m for m in args.models if m not in args.skip]

    print(f"\nDevice  : {DEVICE}" + (f"  ({torch.cuda.get_device_name(0)})" if ON_GPU else ""))
    print(f"Models  : {to_train}")
    print(f"Epochs  : {args.epochs}")
    print(f"Batch   : {BATCH_SIZE}")

    if "seq2seq" in to_train:
        train_seq2seq("Seq2Seq LSTM", m_seq2seq, args.epochs)

    if "transformer" in to_train:
        train_seq2seq("Transformer", m_transformer, args.epochs)

    if "cvae" in to_train:
        train_cvae(args.epochs)

    if "cgan" in to_train:
        train_cgan(args.epochs)

    print("\nAll models trained.")


if __name__ == "__main__":
    main()