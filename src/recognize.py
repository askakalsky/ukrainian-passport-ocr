"""
OCR recognition: preprocessed ROI image -> 8-character passport string.

Model: CRNN (CNN + biGRU + 8 position heads).
Input:  grayscale image of any size (resized internally to 48x256).
Output: (string, confidence)  e.g. ("НР430098", 0.97)
"""

from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn as nn


# -- Constants -----------------------------------------------------------------

DIGITS         = list("0123456789")
SERIES_LETTERS = list("АВЕКМНОРСТИЮ")
ALL_CHARS      = DIGITS + SERIES_LETTERS   # 22 classes

N_CHARS = 8
SEQ_H   = 48
SEQ_W   = 256

CONFIDENCE_THRESHOLD = 0.50   # below this -> flagged as unreadable


# -- Model architecture (must match train/train_sequence.py) -------------------

class SequenceCNN(nn.Module):
    RNN_SLOTS  = 24
    CHAR_SLOTS = [1, 4, 7, 10, 13, 16, 19, 22]

    def __init__(self, n_classes: int = len(ALL_CHARS)):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
        )
        self.pool  = nn.AdaptiveAvgPool2d((1, self.RNN_SLOTS))
        self.rnn   = nn.GRU(128, 128, num_layers=1, batch_first=True,
                            bidirectional=True)
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(256, 64), nn.ReLU(), nn.Dropout(0.3),
                nn.Linear(64, n_classes),
            )
            for _ in range(N_CHARS)
        ])

    def forward(self, x):
        feat = self.features(x)
        feat = self.pool(feat).squeeze(2)
        feat = feat.permute(0, 2, 1)
        rnn_out, _ = self.rnn(feat)
        return torch.stack(
            [self.heads[i](rnn_out[:, self.CHAR_SLOTS[i], :])
             for i in range(N_CHARS)],
            dim=1)


# -- Load ----------------------------------------------------------------------

def load_recognizer(model_path: str | Path):
    """Load CRNN model from checkpoint. Returns (model, meta_dict)."""
    ck = torch.load(str(model_path), map_location="cpu", weights_only=True)
    n_classes = len(ck["all_chars"])
    model = SequenceCNN(n_classes=n_classes)
    model.load_state_dict(ck["model_state"])
    for m in model.modules():
        if hasattr(m, "training"):
            m.training = False
    return model, ck


# -- Inference -----------------------------------------------------------------

def recognize(image: np.ndarray, model: SequenceCNN,
              checkpoint: dict) -> tuple[str, float]:
    """
    Recognize 8-character passport string from a preprocessed ROI image.

    Args:
        image:      grayscale numpy array (any size, will be resized).
        model:      loaded SequenceCNN.
        checkpoint: dict returned by load_recognizer (contains idx2char etc.).

    Returns:
        (prediction, confidence)
        confidence < CONFIDENCE_THRESHOLD means the image is likely unreadable.
    """
    all_chars  = checkpoint["all_chars"]
    idx2char   = {int(k): v for k, v in checkpoint["idx2char"].items()}
    seq_h      = checkpoint.get("seq_h", SEQ_H)
    seq_w      = checkpoint.get("seq_w", SEQ_W)

    letter_set = set(SERIES_LETTERS)
    digit_set  = set(DIGITS)
    letter_idx = [i for i, c in enumerate(all_chars) if c in letter_set]
    digit_idx  = [i for i, c in enumerate(all_chars) if c in digit_set]

    img = cv2.resize(image, (seq_w, seq_h), interpolation=cv2.INTER_AREA)
    arr = img.astype(np.float32) / 255.0
    x   = torch.tensor(arr).unsqueeze(0).unsqueeze(0)   # (1, 1, H, W)

    with torch.no_grad():
        logits = model(x)[0]   # (8, n_classes)

    result, confidences = [], []
    for pos in range(N_CHARS):
        mask = torch.full((len(all_chars),), float("-inf"))
        for idx in (letter_idx if pos < 2 else digit_idx):
            mask[idx] = 0.0
        masked = logits[pos] + mask
        probs  = torch.softmax(masked, dim=0)
        best   = probs.argmax().item()
        result.append(idx2char[best])
        confidences.append(probs[best].item())

    return "".join(result), float(np.mean(confidences))
