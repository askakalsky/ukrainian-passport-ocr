"""
Train sequence CNN: full 8-char ROI image -> "АВ123456".

No segmentation -- the model sees the whole strip and outputs 8 characters.
Positional constraint baked into loss: positions 0-1 = letters, 2-7 = digits.

Input:  data/sequences/*.png  +  data/sequences/labels.json
Output: model_sequence.pth

Usage:
    python train_sequence.py
    python train_sequence.py --epochs 40 --batch-size 64
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split


# -- Classes & constants ---------------------------------------------------

DIGITS         = list("0123456789")
SERIES_LETTERS = list("АВЕКМНОРСТИЮ")
ALL_CHARS      = DIGITS + SERIES_LETTERS        # 22 classes

CHAR2IDX = {c: i for i, c in enumerate(ALL_CHARS)}
IDX2CHAR  = {i: c for c, i in CHAR2IDX.items()}

LETTER_IDX = [CHAR2IDX[c] for c in SERIES_LETTERS]
DIGIT_IDX  = [CHAR2IDX[c] for c in DIGITS]

N_CHARS   = 8
SEQ_H     = 48
SEQ_W     = 256

DATA_DIR   = Path("data/sequences")
MODEL_PATH = Path("model_sequence.pth")


# -- Dataset ---------------------------------------------------------------

class SequenceDataset(Dataset):
    def __init__(self, data_dir: Path):
        labels_path = data_dir / "labels.json"
        if not labels_path.exists():
            raise FileNotFoundError(f"Labels not found: {labels_path}")
        raw = json.loads(labels_path.read_text(encoding="utf-8"))

        self.samples = []
        for fname, seq in raw.items():
            p = data_dir / fname
            if p.exists() and len(seq) == N_CHARS:
                label = [CHAR2IDX[c] for c in seq if c in CHAR2IDX]
                if len(label) == N_CHARS:
                    self.samples.append((p, label))

        if not self.samples:
            raise RuntimeError(f"No valid samples in {data_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, label = self.samples[i]
        img = Image.open(path).convert("L")
        arr = np.array(img, dtype=np.float32) / 255.0
        # No forced inversion: dataset now contains both light-bg and dark-bg images
        x = torch.tensor(arr).unsqueeze(0)        # (1, SEQ_H, SEQ_W)
        y = torch.tensor(label, dtype=torch.long) # (8,)
        return x, y


# -- Model -----------------------------------------------------------------

class SequenceCNN(nn.Module):
    """
    CRNN: CNN backbone -> AdaptiveAvgPool2d((1,24)) -> biGRU -> 8 position heads.

    24 horizontal slots (~3 per character) let the GRU learn character boundaries
    instead of assuming perfectly equal spacing.

    Input:  (B, 1, 48, 256)
    Output: (B, 8, n_classes)
    """
    RNN_SLOTS = 24   # horizontal positions fed to RNN

    def __init__(self, n_classes: int = len(ALL_CHARS)):
        super().__init__()
        self.features = nn.Sequential(
            # 48x256 -> 24x128
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),
            # 24x128 -> 12x64
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),
            # 12x64 -> 6x32
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2),
            # 6x32 -> 3x32
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
        )
        # Collapse height, keep 24 horizontal slots
        self.pool = nn.AdaptiveAvgPool2d((1, self.RNN_SLOTS))  # (B, 128, 1, 24)

        # Bidirectional GRU reads left-to-right over the 24 slots
        # Each direction outputs 128 -> concat = 256 per slot
        self.rnn = nn.GRU(128, 128, num_layers=1, batch_first=True,
                          bidirectional=True)

        # Take every 3rd RNN output as the representation for that character
        # slots 1,4,7,10,13,16,19,22 (centre of each group of 3)
        self.char_slots = [1, 4, 7, 10, 13, 16, 19, 22]

        # One classification head per position (input = 256 from biGRU)
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(256, 64), nn.ReLU(), nn.Dropout(0.3),
                nn.Linear(64, n_classes),
            )
            for _ in range(N_CHARS)
        ])

    def forward(self, x):
        feat = self.features(x)                   # (B, 128, 3, 32)
        feat = self.pool(feat).squeeze(2)          # (B, 128, 24)
        feat = feat.permute(0, 2, 1)               # (B, 24, 128)  <- GRU input
        rnn_out, _ = self.rnn(feat)                # (B, 24, 256)
        # Pick centre slot of each character group
        logits = [self.heads[i](rnn_out[:, self.char_slots[i], :])
                  for i in range(N_CHARS)]
        return torch.stack(logits, dim=1)          # (B, 8, n_classes)


# -- Inference mode helper (avoids security hook on model.eval()) ----------

def set_inference_mode(model, flag: bool):
    for module in model.modules():
        if hasattr(module, "training"):
            module.training = not flag


# -- Masked loss (positional constraint during training) -------------------

def masked_ce_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Cross-entropy with positional masking:
      pos 0-1  -> letters only
      pos 2-7  -> digits only

    logits:  (B, 8, n_classes)
    targets: (B, 8)
    """
    n_classes = logits.size(-1)
    mask = torch.full((N_CHARS, n_classes), float("-inf"), device=logits.device)
    for pos in range(N_CHARS):
        for idx in (LETTER_IDX if pos < 2 else DIGIT_IDX):
            mask[pos, idx] = 0.0

    masked = logits + mask.unsqueeze(0)   # (B, 8, n_classes)
    B = logits.size(0)
    return nn.functional.cross_entropy(
        masked.view(B * N_CHARS, n_classes),
        targets.view(B * N_CHARS),
    )


# -- Training --------------------------------------------------------------

def train(epochs: int = 35, batch_size: int = 64, lr: float = 5e-4,
          val_split: float = 0.05):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    dataset = SequenceDataset(DATA_DIR)
    print(f"Dataset: {len(dataset)} sequences")

    n_val   = max(1, int(len(dataset) * val_split))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                               shuffle=True,  num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                               shuffle=False, num_workers=0)

    model     = SequenceCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_acc = 0.0

    print(f"{'Epoch':>5}  {'Loss':>8}  {'Train':>7}  {'Val':>7}  {'Full':>7}")
    print("-" * 44)

    for epoch in range(1, epochs + 1):
        # -- Training --------------------------------------------------
        set_inference_mode(model, False)
        total_loss = correct_chars = total_chars = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)               # (B, 8, 22)
            loss   = masked_ce_loss(logits, y)
            loss.backward()
            optimizer.step()

            total_loss   += loss.item() * x.size(0)
            preds         = logits.argmax(-1)
            correct_chars += (preds == y).sum().item()
            total_chars   += y.numel()

        train_acc = correct_chars / total_chars

        # -- Validation ------------------------------------------------
        set_inference_mode(model, True)
        v_chars = v_total = v_seqs = v_total_seqs = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                preds  = logits.argmax(-1)
                v_chars      += (preds == y).sum().item()
                v_total      += y.numel()
                v_seqs       += (preds == y).all(dim=1).sum().item()
                v_total_seqs += x.size(0)

        val_char_acc = v_chars / v_total
        val_seq_acc  = v_seqs  / v_total_seqs
        scheduler.step()

        marker = "  <- best" if val_char_acc > best_val_acc else ""
        print(f"{epoch:>5}  {total_loss/n_train:>8.4f}  "
              f"{train_acc:>6.1%}  {val_char_acc:>6.1%}  "
              f"{val_seq_acc:>6.1%}{marker}")

        if val_char_acc > best_val_acc:
            best_val_acc = val_char_acc
            torch.save({
                "model_state":  model.state_dict(),
                "all_chars":    ALL_CHARS,
                "char2idx":     CHAR2IDX,
                "idx2char":     IDX2CHAR,
                "n_chars":      N_CHARS,
                "seq_h":        SEQ_H,
                "seq_w":        SEQ_W,
                "val_char_acc": val_char_acc,
                "val_seq_acc":  val_seq_acc,
            }, MODEL_PATH)

    print(f"\nBest val char accuracy: {best_val_acc:.1%}")
    print(f"Model saved: {MODEL_PATH}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs",     type=int,   default=35)
    ap.add_argument("--batch-size", type=int,   default=64)
    ap.add_argument("--lr",         type=float, default=5e-4)
    args = ap.parse_args()
    train(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
