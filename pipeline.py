"""
Ukrainian passport series/number OCR pipeline.

Full flow:
    passport image  ->  YOLO ROI detection  ->  preprocessing  ->  CRNN OCR

Usage:
    # Python API
    from pipeline import PassportOCR
    ocr = PassportOCR()
    result = ocr("passport.jpg")
    # {"series": "НР", "number": "430098", "full": "НР430098",
    #  "confidence": 0.97, "readable": True}

    # CLI
    python pipeline.py passport.jpg
    python pipeline.py passport.jpg --show
"""

from __future__ import annotations
import argparse
from pathlib import Path

import cv2
import numpy as np

from src.detect     import load_detector, detect_roi, crop_roi
from src.preprocess import preprocess
from src.recognize  import load_recognizer, recognize, CONFIDENCE_THRESHOLD

MODELS_DIR    = Path(__file__).parent / "models"
DETECTOR_PATH = MODELS_DIR / "detector.pt"
RECOGNIZER_PATH = MODELS_DIR / "recognizer.pth"


class PassportOCR:
    """
    End-to-end passport series/number recognizer.

    Args:
        detector_path:   path to YOLO .pt weights.
        recognizer_path: path to CRNN .pth checkpoint.
        det_conf:        YOLO confidence threshold (lower = more sensitive).
    """

    def __init__(
        self,
        detector_path:   str | Path = DETECTOR_PATH,
        recognizer_path: str | Path = RECOGNIZER_PATH,
        det_conf:        float      = 0.30,
    ):
        self.detector,  = load_detector(detector_path),
        self.det_conf   = det_conf
        self.model, self.ckpt = load_recognizer(recognizer_path)

    def __call__(self, image: str | Path | np.ndarray) -> dict:
        """
        Run full pipeline on a passport image.

        Args:
            image: file path (str/Path) or BGR numpy array.

        Returns dict with keys:
            series     — 2 Cyrillic letters  (e.g. "НР")
            number     — 6 digits            (e.g. "430098")
            full       — series + number     (e.g. "НР430098")
            confidence — avg softmax prob    (0.0 – 1.0)
            readable   — False if confidence < threshold
            roi_box    — (x1,y1,x2,y2) or None if detection failed
        """
        # -- Load image --------------------------------------------------------
        if isinstance(image, (str, Path)):
            img_bgr = cv2.imread(str(image))
            if img_bgr is None:
                raise FileNotFoundError(f"Cannot read image: {image}")
        else:
            img_bgr = image

        # -- Detect ROI --------------------------------------------------------
        box = detect_roi(img_bgr, self.detector, conf=self.det_conf)

        if box is None:
            return {
                "series":     None,
                "number":     None,
                "full":       None,
                "confidence": 0.0,
                "readable":   False,
                "roi_box":    None,
                "error":      "ROI not detected",
            }

        roi_crop = crop_roi(img_bgr, box, padding=0.05)

        # -- Preprocess --------------------------------------------------------
        roi_clean = preprocess(roi_crop)

        # -- Recognize ---------------------------------------------------------
        text, conf = recognize(roi_clean, self.model, self.ckpt)

        return {
            "series":     text[:2],
            "number":     text[2:],
            "full":       text,
            "confidence": round(conf, 4),
            "readable":   conf >= CONFIDENCE_THRESHOLD,
            "roi_box":    box,
        }


# -- CLI -----------------------------------------------------------------------

def _show_result(img_bgr: np.ndarray, result: dict) -> None:
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt

    box = result.get("roi_box")
    annotated = img_bgr.copy()
    if box:
        x1, y1, x2, y2 = box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 3)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].imshow(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Detected ROI", fontsize=10)
    axes[0].axis("off")

    if box:
        roi = crop_roi(img_bgr, box, padding=0.05)
        roi_clean = preprocess(roi)
        axes[1].imshow(roi_clean, cmap="gray")
        label = result.get("full") or "NOT DETECTED"
        conf  = result.get("confidence", 0)
        axes[1].set_title(f"{label}  (conf {conf:.1%})", fontsize=12)
        axes[1].axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Ukrainian passport OCR")
    ap.add_argument("image", help="Path to passport image")
    ap.add_argument("--show", action="store_true", help="Show visual result")
    ap.add_argument("--det-conf", type=float, default=0.30,
                    help="YOLO detection confidence threshold")
    args = ap.parse_args()

    ocr    = PassportOCR(det_conf=args.det_conf)
    result = ocr(args.image)

    if result.get("error"):
        print(f"ERROR: {result['error']}")
    else:
        status = "OK" if result["readable"] else "LOW CONFIDENCE"
        print(f"Result:     {result['full']}")
        print(f"Series:     {result['series']}")
        print(f"Number:     {result['number']}")
        print(f"Confidence: {result['confidence']:.1%}  [{status}]")

    if args.show:
        img = cv2.imread(args.image)
        _show_result(img, result)
