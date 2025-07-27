import argparse
import os
import random
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

# ==== hyper‑params (tweak to taste) ====
EMBED_DIM: int = 256
NUM_FILTERS: int = 128
KERNEL_SIZES: list[int] = [2, 3, 4, 5]
DROP_RATE: float = 0.30

MASK_P: float = 0.10          # chance we yeet a token
SPAN_P: float = 0.10          # how much of a span to blank out

GAMMA_FOCAL: float = 2.0
ALPHA_FOCAL: Tuple[float, float, float] = (0.20, 0.60, 0.20)  # neutral gets a smidge more love
SMOOTH_EPS: float = 0.05      # tiny bit of label smoothing

LR: float = 3e‑4              # learning rate, spicy but not too spicy
WD: float = 1e‑5              # weight decay to keep things tidy
MAX_GRAD_NORM: float = 1.0    # gradient clipping cap
NUM_CLASSES: int = 3

# ==== device setup + speed hacks ====
DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True           # let cuDNN pick fastest kernels
torch.set_num_threads(os.cpu_count() or 1)      # max out CPU threads

VOCAB_SIZE: int = 10  # will get overwritten once we peek at the data

# ---------- dataset with augmentations and essentially powerups to kinda boost acc-----------
class EmotionAugDataset(Dataset):
    """Wraps the numpy arrays and sprinkles in some noise if asked."""

    def __init__(
        self,
        sentences: Sequence[Sequence[int]],
        labels: Sequence[int] | None = None,
        augment: bool = False,
    ) -> None:
        self.sentences = sentences
        self.labels = labels
        self.augment = augment

    def __len__(self) -> int:
        return len(self.sentences)

    # a couple of grubby helpers
    @staticmethod
    def _token_drop(tokens: list[int]) -> list[int]:
        kept = [t for t in tokens if random.random() > MASK_P]
        return kept or tokens  # avoid empty inputs

    @staticmethod
    def _span_mask(tokens: list[int]) -> list[int]:
        if random.random() >= 0.5:
            return tokens
        span_len = max(1, int(len(tokens) * SPAN_P))
        start = random.randint(0, len(tokens) - span_len)
        tokens[start : start + span_len] = [0] * span_len
        return tokens

    def _augment(self, seq: Sequence[int]) -> list[int]:
        tokens = list(seq)
        tokens = self._token_drop(tokens)
        tokens = self._span_mask(tokens)
        return tokens

    def __getitem__(self, idx: int):
        seq = self.sentences[idx]
        if self.augment:
            seq = self._augment(seq)

        x = torch.tensor(seq, dtype=torch.long)
        if self.labels is None:
            return x
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y


def collate(batch):
    """Pads things so the GPU isn’t confused."""
    if isinstance(batch[0], tuple):
        seqs, labels = zip(*batch)
        x_pad = nn.utils.rnn.pad_sequence(seqs, batch_first=True, padding_value=0)
        return x_pad.to(DEVICE), torch.stack(labels).to(DEVICE)
    x_pad = nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0)
    return x_pad.to(DEVICE)


# ---------- CNN(why? Because this is a NLP + CNN since the data is 1D array so i thought it would help)------------
class TextCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(VOCAB_SIZE + 2, EMBED_DIM, padding_idx=0)
        self.convs = nn.ModuleList(
            nn.Conv1d(EMBED_DIM, NUM_FILTERS, k, padding=k // 2)
            for k in KERNEL_SIZES
        )
        self.dropout = nn.Dropout(DROP_RATE)
        self.classifier = nn.Linear(NUM_FILTERS * len(KERNEL_SIZES), NUM_CLASSES)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x).transpose(1, 2)               # (B, E, T)
        pooled = [torch.relu(c(x)).max(dim=2).values for c in self.convs]
        return self.classifier(self.dropout(torch.cat(pooled, dim=1)))


# ---------- spicy focal + smooth loss ----------
class FocalLabelSmoothingLoss(nn.Module):
    def __init__(
        self,
        alpha: Tuple[float, float, float] = ALPHA_FOCAL,
        gamma: float = GAMMA_FOCAL,
        eps: float = SMOOTH_EPS,
    ) -> None:
        super().__init__()
        self.alpha = torch.tensor(alpha)
        self.gamma = gamma
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        n_cls = logits.size(1)
        smooth = torch.full_like(logits, self.eps / (n_cls - 1))
        smooth.scatter_(1, targets.unsqueeze(1), 1 - self.eps)

        log_probs = torch.log_softmax(logits, dim=1)
        probs = log_probs.exp()
        pt = torch.sum(probs * smooth, dim=1)

        alpha_t = self.alpha.to(logits.device)[targets]
        loss = -alpha_t * (1 - pt) ** self.gamma * torch.sum(log_probs * smooth, dim=1)
        return loss.mean()


criterion = FocalLabelSmoothingLoss()

# ---------- one epoch of grind ----------
def train_epoch(model: nn.Module, loader: DataLoader, opt: optim.Optimizer) -> None:
    model.train()
    for xb, yb in loader:
        opt.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
        opt.step()


@torch.inference_mode()
def evaluate(model: nn.Module, loader: DataLoader) -> float:
    model.eval()
    preds: list[int] = []
    gold: list[int] = []

    for xb, yb in loader:
        preds.extend(model(xb).argmax(1).cpu().tolist())
        gold.extend(yb.cpu().tolist())

    f1 = f1_score(gold, preds, average="macro")
    cm = confusion_matrix(gold, preds, labels=list(range(NUM_CLASSES)))

    print("\n⚔️  Confusion matrix")
    print(
        pd.DataFrame(
            cm,
            index=["true_neg", "true_neu", "true_pos"],
            columns=["pred_neg", "pred_neu", "pred_pos"],
        )
    )
    print("\n Classification report")
    print(classification_report(gold, preds, target_names=["neg", "neu", "pos"]))
    return f1


def make_loader(
    ds: Dataset,
    batch_size: int,
    *,
    sampler=None,
    shuffle: bool = False,
) -> DataLoader:
    return DataLoader(
        ds,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=shuffle if sampler is None else False,
        num_workers=min(8, os.cpu_count() or 1),
        pin_memory=(DEVICE.type == "cuda"),
        prefetch_factor=4,
        persistent_workers=True,
        collate_fn=collate,
    )


# ---------- the whole shebang ----------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=Path,
        default="~/vietnam-machine-learning-project/data",
        help="Folder with those juicy *.npy files",
    )
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    data_dir = Path(args.data_dir).expanduser()
    train_X = np.load(data_dir / "train_X.npy", allow_pickle=True)
    train_Y = np.load(data_dir / "train_Y.npy", allow_pickle=True)
    public_X = np.load(data_dir / "public_X.npy", allow_pickle=True)
    out_path = data_dir / "predictions_public.npy"

    global VOCAB_SIZE
    VOCAB_SIZE = int(max(map(max, train_X)))

    ensemble_logits: list[np.ndarray] = []

    for seed in range(args.seeds):
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # split 90/10 (because why not)
        tr_X, val_X, tr_y, val_y = train_test_split(
            train_X,
            train_Y,
            test_size=0.1,
            stratify=train_Y,
            random_state=seed,
        )

        # give rare classes a leg up
        counts = np.bincount(tr_y)
        weights = 1.0 / counts
        weights[1] *= 2.0
        sampler = WeightedRandomSampler(weights[tr_y], len(tr_y), replacement=True)

        train_ld = make_loader(
            EmotionAugDataset(tr_X, tr_y, augment=True),
            batch_size=args.batch_size,
            sampler=sampler,
        )
        val_ld = make_loader(
            EmotionAugDataset(val_X, val_y),
            batch_size=args.batch_size,
        )

        model = TextCNN().to(DEVICE)
        opt = optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)

        for _ in range(args.epochs):
            train_epoch(model, train_ld, opt)

        val_f1 = evaluate(model, val_ld)
        print(f"[seed {seed}] val macro‑F1: {val_f1:.4f}")

        # now throw the full dataset at it
        full_sampler = WeightedRandomSampler(
            weights[train_Y], len(train_Y), replacement=True
        )
        full_ld = make_loader(
            EmotionAugDataset(train_X, train_Y, augment=True),
            batch_size=args.batch_size,
            sampler=full_sampler,
        )

        model_full = TextCNN().to(DEVICE)
        opt_full = optim.AdamW(model_full.parameters(), lr=LR, weight_decay=WD)

        for _ in range(args.epochs):
            train_epoch(model_full, full_ld, opt_full)

        # grab logits on the public set
        pub_ld = make_loader(EmotionAugDataset(public_X), batch_size=args.batch_size)

        with torch.inference_mode():
            logits_seed: list[np.ndarray] = []
            for xb in pub_ld:
                logits_seed.append(torch.softmax(model_full(xb), dim=1).cpu().numpy())
            ensemble_logits.append(np.vstack(logits_seed))

    # mash the seeds together
    preds = np.mean(ensemble_logits, axis=0).argmax(axis=1)
    np.save(out_path, preds.astype(np.int64))
    print(" Saved predictions to", out_path)


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn", force=True)  # macOS grumbles otherwise
    main()
