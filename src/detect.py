"""
ROI detection: full passport image -> bounding box of the dotted series/number strip.
Uses a YOLO model fine-tuned on Ukrainian passport images (class: "Dotted").
"""

from pathlib import Path
import cv2
import numpy as np


def load_detector(model_path: str | Path):
    from ultralytics import YOLO
    return YOLO(str(model_path))


def detect_roi(image: np.ndarray, model, conf: float = 0.30) -> tuple[int, int, int, int] | None:
    """
    Detect the dotted passport series/number strip in a full passport image.

    Args:
        image:  BGR image (numpy array) or grayscale — will be converted as needed.
        model:  loaded YOLO model (from load_detector).
        conf:   minimum confidence threshold.

    Returns:
        (x1, y1, x2, y2) pixel coordinates of the best detection,
        or None if nothing found.
    """
    results = model(image, conf=conf, verbose=False)
    boxes = results[0].boxes

    if boxes is None or len(boxes) == 0:
        return None

    # Pick highest-confidence box
    best = boxes[boxes.conf.argmax()]
    x1, y1, x2, y2 = map(int, best.xyxy[0].tolist())
    return x1, y1, x2, y2


def crop_roi(image: np.ndarray, box: tuple[int, int, int, int],
             padding: float = 0.05) -> np.ndarray:
    """
    Crop the ROI from the image with optional relative padding.

    Args:
        image:   BGR or grayscale image.
        box:     (x1, y1, x2, y2)
        padding: fraction of box size to add as margin on each side.

    Returns:
        Cropped numpy array (same channels as input).
    """
    h, w = image.shape[:2]
    x1, y1, x2, y2 = box

    pw = int((x2 - x1) * padding)
    ph = int((y2 - y1) * padding)

    x1 = max(0, x1 - pw)
    y1 = max(0, y1 - ph)
    x2 = min(w, x2 + pw)
    y2 = min(h, y2 + ph)

    return image[y1:y2, x1:x2]
