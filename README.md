# Dates Generator

Conditional date generation using four deep generative models. Given a set of conditions (weekday, month, leap year, decade), the model generates a valid date that satisfies all of them.

---

## Problem

**Input:** `[WED] [JAN] [False] [180]`  
**Output:** `1-1-1800` *(any valid date matching all 4 conditions)*

---

## Models

| # | Model | Type | Condition Accuracy |
|---|-------|------|--------------------|
| 1 | Seq2Seq LSTM + Attention | Course | ~16–18% |
| 2 | Transformer Encoder-Decoder | Outside course | ~15–17% |
| 3 | Conditional VAE (CVAE) | Outside course | ~18% |
| 4 | Conditional GAN (cGAN) | Course (required) | ~0%* |

*GANs cannot backpropagate through discrete token selection (argmax is non-differentiable) — a known fundamental limitation.

---

## Repo Structure

```
Assignment 2/
├── data/
│   ├── data.txt               # full dataset
│   └── example_input.txt      # input-only file for inference
├── model/
│   ├── train_all.py           # train all 4 models with one command
│   ├── predict.py             # inference script
│   ├── dataset.py             # shared dataset
│   ├── utils.py               # vocabulary, parsing, validation
│   ├── model_seq2seq.py       # Model 1
│   ├── model_transformer.py   # Model 2
│   ├── model_cvae.py          # Model 3
│   ├── model_cgan.py          # Model 4
│   ├── seq2seq_lstm.pth       # trained weights
│   ├── transformer.pth
│   ├── cvae.pth
│   └── cgan.pth
└── Assignment_2_Mohamed_Emam.docx
```

---

## Setup

```bash
python -m venv venv
venv\Scripts\activate        # Windows
pip install torch torchvision
```

---

## Train

```bash
cd model

# Train all 4 models
python train_all.py

# Train specific models only
python train_all.py --models transformer cvae

# Skip already-trained models
python train_all.py --skip seq2seq
```

---

## Inference

```bash
# Default model (seq2seq — best accuracy)
python predict.py -i ../data/example_input.txt -o ../data/output.txt

# Choose a specific model
python predict.py -i ../data/example_input.txt -o ../data/output.txt --model transformer
python predict.py -i ../data/example_input.txt -o ../data/output.txt --model cvae
python predict.py -i ../data/example_input.txt -o ../data/output.txt --model cgan
```

Output format matches `data.txt` exactly:
```
[WED] [JAN] [False] [180] 1-1-1800
[MON] [JAN] [False] [190] 2-1-1900
```

---

## Key Design Decisions

- **Year-first token order (`yyyy-mm-dd`)** — generating year digits first aligns the output with the decade/leap-year constraints, where gradient signal is strongest. This was the single biggest accuracy improvement (0% → 16%).
- **Bahdanau attention** — lets the decoder re-query each condition token at every generation step instead of compressing all conditions into one fixed vector.
- **KL annealing in CVAE** — β ramps from 0 → 1 over 20 epochs to prevent posterior collapse.
- **Teacher-forcing annealing** — starts at 50%, decays to 10% to close the train/inference gap.
