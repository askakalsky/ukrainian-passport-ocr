"""
Preprocessing pipeline: raw ROI crop -> clean grayscale binary image.

Steps (tuned for Ukrainian dot-matrix passport series/number strips):
  1. Grayscale
  2. Non-local means denoising
  3. (Optional) CLAHE / top-hat / sharpening
  4. (Optional) Gamma correction
  5. Otsu binarisation
"""

import cv2
import numpy as np


def preprocess(image: np.ndarray) -> np.ndarray:
    """
    Full preprocessing pipeline.

    Args:
        image: BGR or grayscale crop of the passport ROI.

    Returns:
        uint8 grayscale image after binarisation (0 = background, 255 = ink).
    """
    # 1. Grayscale
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # 2. Non-local means denoising
    denoised = cv2.fastNlMeansDenoising(gray, h=3, templateWindowSize=9,
                                        searchWindowSize=5)

    # 3. Enhancement (all disabled by default — add if needed per scan quality)
    enhanced = denoised

    # 4. Gamma correction (disabled — gamma=1.0 is identity)
    # gamma = 1.8
    # lut = np.array([((i / 255.0) ** (1.0 / gamma)) * 255
    #                 for i in range(256)], dtype=np.uint8)
    # enhanced = cv2.LUT(enhanced, lut)

    # 5. Otsu binarisation
    _, binary = cv2.threshold(enhanced, 0, 255,
                               cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return binary
