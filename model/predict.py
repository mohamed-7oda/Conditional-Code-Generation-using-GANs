"""
predict.py  –  Run inference with any of the four trained models.

Usage:
    python predict.py -i data/example_input.txt -o output.txt
    python predict.py -i data/example_input.txt -o output.txt --model transformer
    python predict.py -i data/example_input.txt -o output.txt --model cvae
    python predict.py -i data/example_input.txt -o output.txt --model cgan

Default model is seq2seq (Model 1 — best performing).
"""

import argparse
from pathlib import Path

import torch

from utils import build_vocab, parse_line, tokens_to_date
import model_seq2seq    as m_seq2seq
import model_transformer as m_transformer
import model_cvae        as m_cvae
import model_cgan        as m_cgan

char2idx, idx2char = build_vocab()

MODEL_REGISTRY = {
    "seq2seq":     m_seq2seq,
    "transformer": m_transformer,
    "cvae":        m_cvae,
    "cgan":        m_cgan,
}


def load_model(model_name: str, device: torch.device):
    module = MODEL_REGISTRY[model_name]
    if not module.WEIGHTS_PATH.exists():
        raise FileNotFoundError(
            f"No weights found for '{model_name}' at {module.WEIGHTS_PATH}.\n"
            f"Run: python train_all.py --models {model_name}"
        )
    return module.load_checkpoint(module.WEIGHTS_PATH, device)


def predict_batch(model, cond_list: list[list[str]], device: torch.device) -> list[str]:
    src = torch.tensor(
        [[char2idx[t] for t in cond] for cond in cond_list],
        dtype=torch.long, device=device,
    )
    all_ids = model.generate_batch(src)
    return [tokens_to_date([idx2char[i] for i in ids]) for ids in all_ids]


def format_line(cond_tokens: list[str], date_str: str) -> str:
    day, month, leap, decade = cond_tokens
    return f"[{day}] [{month}] [{leap}] [{decade}] {date_str}"


def run(input_path: Path, output_path: Path, model_name: str, batch_size: int = 256) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Model : {model_name}  |  Device: {device}")

    model = load_model(model_name, device)
    model.eval()

    with open(input_path) as f:
        lines = [l.strip() for l in f if l.strip()]

    all_conds   = []
    skip_idx    = set()
    for i, line in enumerate(lines):
        try:
            cond_toks, _ = parse_line(line)
            all_conds.append(cond_toks)
        except ValueError as e:
            print(f"  Warning line {i+1}: {e}")
            skip_idx.add(i)
            all_conds.append(None)

    results: list[str] = []
    valid_conds = [(i, c) for i, c in enumerate(all_conds) if c is not None]

    for start in range(0, len(valid_conds), batch_size):
        batch = valid_conds[start: start + batch_size]
        idxs, conds = zip(*batch)
        dates = predict_batch(model, list(conds), device)
        for idx, cond, date in zip(idxs, conds, dates):
            results.append((idx, format_line(cond, date)))
        if (start + batch_size) % (batch_size * 5) == 0:
            print(f"  Processed {start + batch_size}/{len(valid_conds)} ...")

    results.sort(key=lambda x: x[0])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(line for _, line in results) + "\n")

    print(f"Done. Wrote {len(results)} predictions → {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Date Generator — inference")
    parser.add_argument("-i", "--input",  required=True, type=Path)
    parser.add_argument("-o", "--output", required=True, type=Path)
    parser.add_argument("--model", default="seq2seq", choices=list(MODEL_REGISTRY.keys()),
                        help="Which model to use (default: seq2seq)")
    args = parser.parse_args()
    run(args.input, args.output, args.model)


if __name__ == "__main__":
    main()