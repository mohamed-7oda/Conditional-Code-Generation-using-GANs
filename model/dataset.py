"""
dataset.py  –  Shared DateDataset and collate_fn used by all four models.
"""

from pathlib import Path

import torch
from torch.utils.data import Dataset

from utils import build_vocab, parse_line, date_to_tokens, PAD_IDX, SOS_IDX, EOS_IDX

char2idx, idx2char = build_vocab()


class DateDataset(Dataset):
    """Each sample: (condition_ids [4], target_ids [variable length])."""

    def __init__(self, filepath: Path) -> None:
        self.samples: list[tuple[list[int], list[int]]] = []
        with open(filepath) as f:
            for line in f:
                try:
                    cond_toks, date_str = parse_line(line)
                except ValueError:
                    continue
                if date_str is None:
                    continue
                cond_ids = [char2idx[t] for t in cond_toks]
                date_char_toks = date_to_tokens(date_str)
                date_ids = [SOS_IDX] + [char2idx[c] for c in date_char_toks] + [EOS_IDX]
                self.samples.append((cond_ids, date_ids))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[list[int], list[int]]:
        return self.samples[idx]


def collate_fn(
    batch: list[tuple[list[int], list[int]]]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    conds, targets = zip(*batch)
    cond_tensor = torch.tensor(conds, dtype=torch.long)
    max_t = max(len(t) for t in targets)
    padded = [t + [PAD_IDX] * (max_t - len(t)) for t in targets]
    target_tensor = torch.tensor(padded, dtype=torch.long)
    lengths = torch.tensor([len(t) for t in targets], dtype=torch.long)
    return cond_tensor, target_tensor, lengths