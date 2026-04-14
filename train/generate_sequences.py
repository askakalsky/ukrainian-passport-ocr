"""
Generate synthetic 8-character passport sequence images for CRNN training.

Each image is a full ROI strip (2 letters + 6 digits) rendered side by side
from the dot-matrix font, matching real passport proportions (~5:1 width:height).

Output:
    data/sequences/         PNG images (48 x 256 grayscale)
    data/sequences/labels.json   { "filename.png": "АВ123456", ... }

Usage:
    python generate_sequences.py --n-samples 50000
    python generate_sequences.py --n-samples 200 --show-samples
"""

import argparse
import json
import random
import uuid
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageFilter


# ── Config ─────────────────────────────────────────────────────────────

SERIES_LETTERS = list("АВЕКМНОРСТИЮ")
DIGITS         = list("0123456789")
N_CHARS        = 8    # 2 letters + 6 digits

SEQ_H = 48            # training image height (px)
SEQ_W = 256           # training image width  (px)

OUT_DIR = Path("data/sequences")


# ── Font loading (reused from generate_from_font.py) ───────────────────

def load_font() -> dict:
    import json as _json
    custom = Path("font_custom.json")
    if custom.exists():
        raw = _json.loads(custom.read_text(encoding="utf-8"))
        result = {}
        all_chars = set(DIGITS + SERIES_LETTERS)
        for ch, rows in raw.items():
            if ch in all_chars:
                result[ch] = np.array(rows, dtype=np.uint8)
        if result:
            return result
    from font5x7_cyrillic import FONT
    all_chars = set(DIGITS + SERIES_LETTERS)
    return {ch: grid for ch, grid in FONT.items() if ch in all_chars}


# ── Sequence renderer ──────────────────────────────────────────────────

def render_sequence(seq: str, font_data: dict,
                    jitter: bool = True,
                    canvas_h: int = SEQ_H,
                    canvas_w: int = SEQ_W) -> np.ndarray:
    """
    Render an 8-character sequence on a (canvas_h x canvas_w) canvas.

    Characters are placed side by side with small inter-char gaps,
    centred both horizontally and vertically.
    Returns float32 array, 0=ink, 255=paper.
    """
    bg = random.uniform(218, 248) if jitter else 235.0
    canvas = np.full((canvas_h, canvas_w), bg, dtype=np.float32)

    # ── Compute layout ───────────────────────────────────────────────
    char_rows = 6                    # all font chars are 6 rows tall
    char_ws   = [font_data[c].shape[1] for c in seq]  # dot columns per char

    # Gap between chars: random fraction of 1 dot column
    gap_dots = random.uniform(0.3, 1.2) if jitter else 0.7

    total_dots_w = sum(char_ws) + gap_dots * (len(seq) - 1)

    # Cell size: largest that fits both axes
    cell_from_h = canvas_h * random.uniform(0.60, 0.80) / char_rows  \
                  if jitter else canvas_h * 0.70 / char_rows
    cell_from_w = canvas_w * random.uniform(0.82, 0.92) / total_dots_w \
                  if jitter else canvas_w * 0.88 / total_dots_w
    cell = min(cell_from_h, cell_from_w)

    gap_px      = gap_dots * cell
    render_w    = sum(cw * cell for cw in char_ws) + gap_px * (len(seq) - 1)
    render_h    = char_rows * cell

    # Random vertical/horizontal offset within the canvas
    max_dx = max(0, (canvas_w - render_w) * 0.4)
    max_dy = max(0, (canvas_h - render_h) * 0.4)
    off_x = (canvas_w - render_w) / 2.0 + (random.uniform(-max_dx, max_dx) if jitter else 0)
    off_y = (canvas_h - render_h) / 2.0 + (random.uniform(-max_dy, max_dy) if jitter else 0)

    # Dot style: solid or donut (white center, like real perforations)
    donut      = jitter and random.random() < 0.45
    hole_frac  = random.uniform(0.30, 0.55) if donut else 0.0
    # Dot size: vary smaller to larger
    r_factor   = random.uniform(0.22, 0.40) if jitter else 0.35

    # ── Draw each character ──────────────────────────────────────────
    x_cur = off_x
    for ch in seq:
        grid = font_data[ch]
        H, W = grid.shape
        base_r = cell * r_factor
        radius = max(1.5, base_r)

        for r in range(H):
            for c in range(W):
                if grid[r, c] == 0:
                    continue
                cx = x_cur + (c + 0.5) * cell
                cy = off_y + (r + 0.5) * cell
                ink = random.uniform(10, 60) if jitter else 30.0
                icx, icy = int(round(cx)), int(round(cy))
                # Outer dot
                cv2.circle(canvas, (icx, icy), int(round(radius)),
                           ink, -1, lineType=cv2.LINE_AA)
                # Inner hole (donut effect)
                if donut and hole_frac > 0:
                    hole_r = max(1, int(round(radius * hole_frac)))
                    hole_brightness = random.uniform(bg * 0.85, bg * 1.05) if jitter else bg
                    cv2.circle(canvas, (icx, icy), hole_r,
                               min(255, hole_brightness), -1, lineType=cv2.LINE_AA)
        x_cur += W * cell + gap_px

    return canvas


# ── Augmentation (whole-strip, same functions as generate_from_font.py) ─

def aug_perspective(img: np.ndarray, strength: float = 0.07) -> np.ndarray:
    h, w = img.shape
    d = max(1, int(min(h, w) * strength))
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    def jit(): return random.randint(-d, d)
    dst = np.float32([
        [jit(), jit()], [w + jit(), jit()],
        [w + jit(), h + jit()], [jit(), h + jit()],
    ])
    M = cv2.getPerspectiveTransform(src, dst)
    fill = float(img[0, 0])
    return cv2.warpPerspective(img, M, (w, h),
                               borderMode=cv2.BORDER_CONSTANT,
                               borderValue=fill).astype(np.float32)


def aug_rotation(img: np.ndarray, angle: float) -> np.ndarray:
    h, w = img.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    fill = float(img[0, 0])
    return cv2.warpAffine(img, M, (w, h),
                          borderMode=cv2.BORDER_CONSTANT,
                          borderValue=fill).astype(np.float32)


def aug_shadow_stripe(img: np.ndarray, prob: float = 0.40) -> np.ndarray:
    """
    Bright horizontal band at top and/or bottom (light reflection / shadow edge).
    Applied AFTER rotation so the stripe is always horizontal in the frame.
    """
    if random.random() > prob:
        return img
    out = img.astype(np.float32).copy()
    h, w = out.shape
    sides = random.choice(['top', 'bottom', 'both'])
    for side in (['top', 'bottom'] if sides == 'both' else [sides]):
        size   = int(h * random.uniform(0.15, 0.45))
        bright = random.uniform(210, 255)
        for row in range(size):
            if side == 'top':
                alpha = 1.0 - row / size
                r = row
            else:
                alpha = row / size
                r = h - size + row
            out[r, :] = out[r, :] * (1 - alpha) + bright * alpha
    return np.clip(out, 0, 255).astype(np.float32)


def aug_illumination(img: np.ndarray) -> np.ndarray:
    h, w = img.shape
    angle = random.uniform(0, 2 * np.pi)
    gx = np.cos(angle) * np.linspace(-1, 1, w)
    gy = np.sin(angle) * np.linspace(-1, 1, h)[:, None]
    grad = 1.0 + (gx + gy) * random.uniform(0.03, 0.20)
    return np.clip(img * grad, 0, 255).astype(np.float32)


def aug_noise(img: np.ndarray) -> np.ndarray:
    sigma = random.uniform(20, 65)
    # Occasional heavy-damage frame
    if random.random() < 0.15:
        sigma = random.uniform(60, 110)
    return np.clip(img + np.random.normal(0, sigma, img.shape).astype(np.float32),
                   0, 255).astype(np.float32)


def aug_speckles(img: np.ndarray, prob: float = 0.55) -> np.ndarray:
    if random.random() > prob:
        return img
    out = img.copy()
    h, w = out.shape
    for _ in range(random.randint(3, 14)):
        cx = random.randint(0, w - 1)
        cy = random.randint(0, h - 1)
        r  = random.randint(1, 8)
        cv2.circle(out, (cx, cy), r, random.uniform(5, 100), -1,
                   lineType=cv2.LINE_AA)
    return out.astype(np.float32)


def aug_blur(img: np.ndarray) -> np.ndarray:
    r = random.uniform(0, 1.2)
    if r > 0.15:
        pil = Image.fromarray(img.astype(np.uint8))
        pil = pil.filter(ImageFilter.GaussianBlur(radius=r))
        return np.array(pil, dtype=np.float32)
    return img


def find_text_bbox(img: np.ndarray, margin: int = 8):
    """
    Find tight bounding box around text dots (dark on light background).
    Works on clean rendered image BEFORE noise is added.
    Returns (x1, y1, x2, y2).
    """
    h, w = img.shape
    mask = img < 160          # text pixels are dark, background is ~220-248
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any() or not cols.any():
        return 0, 0, w, h
    y1 = max(0,     int(np.where(rows)[0][0])  - margin)
    y2 = min(h - 1, int(np.where(rows)[0][-1]) + margin)
    x1 = max(0,     int(np.where(cols)[0][0])  - margin)
    x2 = min(w - 1, int(np.where(cols)[0][-1]) + margin)
    return x1, y1, x2, y2


def full_augment_sequence(seq: str, font_data: dict) -> np.ndarray:
    """
    Render + augment -> uint8 (SEQ_H x SEQ_W).

    Uses a large render canvas + bbox detection to guarantee
    no character is ever clipped, regardless of rotation or perspective.

    Flow:
      1. Render on large canvas (big enough for any 20° rotation).
      2. Apply perspective + rotation (no noise yet — image is clean).
      3. Detect text bounding box by thresholding.
      4. Crop to bbox, resize to SEQ_W x SEQ_H.
      5. Apply photometric augmentation (noise, blur, etc.).
      6. Add horizontal shadow stripe.
      7. Optionally invert (dark bg / light dots).
    """
    # Large canvas: 2x width, 6x height — safe for ±20° rotation of wide strip
    CNAME_W = SEQ_W * 2   # 512 px
    CNAME_H = SEQ_H * 6   # 288 px

    angle = random.uniform(-20.0, 20.0)

    # ── 1. Render on big canvas ────────────────────────────────────────
    img = render_sequence(seq, font_data, jitter=True,
                          canvas_h=CNAME_H, canvas_w=CNAME_W)

    # ── 2. Geometric transforms (clean image) ─────────────────────────
    img = aug_perspective(img, strength=0.05)
    img = aug_rotation(img, angle=angle)

    # ── 3. Find text bbox on still-clean image ────────────────────────
    x1, y1, x2, y2 = find_text_bbox(img, margin=10)
    # Ensure minimum crop size to avoid degenerate resize
    if (x2 - x1) < 16 or (y2 - y1) < 8:
        x1, y1, x2, y2 = 0, 0, CNAME_W, CNAME_H

    crop = img[y1:y2, x1:x2]

    # ── 4. Resize crop to target ──────────────────────────────────────
    result = cv2.resize(crop, (SEQ_W, SEQ_H), interpolation=cv2.INTER_AREA)

    # ── 5. Photometric augmentation ───────────────────────────────────
    result = aug_illumination(result)
    result = aug_noise(result)
    result = aug_speckles(result)
    result = aug_blur(result)
    result = np.clip(result, 0, 255).astype(np.uint8)

    # ── 6. Shadow stripe (always horizontal in final frame) ───────────
    result = aug_shadow_stripe(result.astype(np.float32))
    result = np.clip(result, 0, 255).astype(np.uint8)

    # ── 7. Inversion ──────────────────────────────────────────────────
    if random.random() < 0.40:
        result = 255 - result
    return result


# ── Random string generation ───────────────────────────────────────────

def random_sequence() -> str:
    l1 = random.choice(SERIES_LETTERS)
    l2 = random.choice(SERIES_LETTERS)
    digits = "".join(random.choice(DIGITS) for _ in range(6))
    return l1 + l2 + digits


# ── Main generation loop ───────────────────────────────────────────────

def generate(n_samples: int, show_samples: bool = False):
    font_data = load_font()
    missing = [c for c in SERIES_LETTERS + DIGITS if c not in font_data]
    if missing:
        print(f"WARNING: characters missing from font: {missing}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    labels_path = OUT_DIR / "labels.json"

    # Load existing labels so we can append if needed
    if labels_path.exists():
        existing = json.loads(labels_path.read_text(encoding="utf-8"))
    else:
        existing = {}

    print(f"Generating {n_samples} sequence images -> {OUT_DIR}/")
    new_labels = {}

    for i in range(n_samples):
        seq = random_sequence()
        img = full_augment_sequence(seq, font_data)
        fname = f"seq_{uuid.uuid4().hex}.png"
        cv2.imwrite(str(OUT_DIR / fname), img)
        new_labels[fname] = seq

        if (i + 1) % 5000 == 0 or i == n_samples - 1:
            print(f"  {i + 1}/{n_samples}")

    existing.update(new_labels)
    labels_path.write_text(json.dumps(existing, ensure_ascii=False, indent=None),
                           encoding="utf-8")
    print(f"Labels saved: {labels_path}  ({len(existing)} total entries)")

    if show_samples:
        _show_grid(font_data)


def _show_grid(font_data: dict, n: int = 20):
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt

    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 1.5))
    axes_flat = axes.flatten()

    for idx in range(n):
        seq = random_sequence()
        img = full_augment_sequence(seq, font_data)
        ax = axes_flat[idx]
        ax.imshow(img, cmap="gray", vmin=0, vmax=255)
        ax.set_title(seq, fontsize=8)
        ax.axis("off")
    for idx in range(n, len(axes_flat)):
        axes_flat[idx].axis("off")

    plt.suptitle("Synthetic sequence samples", fontsize=10)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-samples",   type=int, default=50000)
    ap.add_argument("--show-samples", action="store_true")
    args = ap.parse_args()
    generate(args.n_samples, args.show_samples)
