---
language:
  - uk
license: mit
tags:
  - ocr
  - object-detection
  - image-classification
  - passport
  - ukraine
  - crnn
  - yolo
  - dot-matrix
pipeline_tag: image-to-text
---

# Ukrainian Passport Series/Number OCR

Automatic recognition of the series and number from Ukrainian internal passports (старого зразка).

The system reads the dot-matrix perforated 8-character string — **2 Cyrillic letters + 6 digits** — from a full passport photo.

```
НР 430098
^^─────── series  (one of: А В Е К М Н О Р С Т И Ю)
   ^^^^^^ number  (6 digits)
```

---

## Pipeline

```
passport image
      │
      ▼
 YOLO detector          ← finds the dotted strip (class: "Dotted")
      │
      ▼
  preprocessing         ← grayscale → denoise → Otsu binarisation
      │
      ▼
  CRNN recognizer       ← CNN + biGRU + 8 position heads → "НР430098"
      │
      ▼
{"series": "НР", "number": "430098", "confidence": 0.97}
```

---

## Quick start

```bash
pip install -r requirements.txt

python pipeline.py passport.jpg
python pipeline.py passport.jpg --show
```

Output:
```
Result:     НР430098
Series:     НР
Number:     430098
Confidence: 97.3%  [OK]
```

---

## Python API

```python
from pipeline import PassportOCR

ocr    = PassportOCR()
result = ocr("passport.jpg")

print(result["full"])        # "НР430098"
print(result["series"])      # "НР"
print(result["number"])      # "430098"
print(result["confidence"])  # 0.9734
print(result["readable"])    # True
```

---

## Models

| File | Architecture | Size | Purpose |
|------|-------------|------|---------|
| `models/detector.pt` | YOLO nano fine-tuned | 5.95 MB | Locates the dotted strip on the passport page |
| `models/recognizer.pth` | CNN + biGRU (FP32) | 2.25 MB | Reads the 8-character string from the strip |
| `models/recognizer_fp16.pth` | CNN + biGRU (FP16) | **1.13 MB** | Same accuracy, 2× smaller — recommended for deployment |

To use the FP16 model:
```python
ocr = PassportOCR(recognizer_path="models/recognizer_fp16.pth")
```

The recognizer is trained on **80 000 synthetic images** generated with a custom dot-matrix Cyrillic font, with heavy augmentation (rotation ±20°, perspective, noise, donut dots, shadow stripes, inversion).

Val character accuracy: **96.1%** | Full-string accuracy on real passports: **96%**

---

## Training

```bash
# Generate synthetic training data
cd train
python generate_sequences.py --n-samples 80000

# Train the recognizer
python train_sequence.py --epochs 35 --batch-size 64
```

To retrain the YOLO detector, annotate passport images with the "Dotted" class and run standard YOLOv8 training.

---

## Series letters

Valid Ukrainian passport series letters: `А В Е К М Н О Р С Т И Ю`

The model enforces positional constraints: positions 0–1 accept only series letters, positions 2–7 accept only digits.

---

## Custom fonts & datasets

The `train/font_editor.py` tool lets you define your own dot-matrix font for any character set:

- Add / remove any Unicode character
- Set any grid size per character (cols × rows, independently)
- Save to `font_custom.json`, export to `font_custom.py`

```bash
cd train
python font_editor.py                      # edit default font
python font_editor.py --font myfont.json   # open a different font file
```

After editing, regenerate synthetic data and retrain:
```bash
python generate_sequences.py --n-samples 80000
python train_sequence.py --epochs 35
```

---

## Limitations

- Works on Ukrainian **internal** passports (старого зразка, до 2016 року)
- Requires the dot-matrix strip to be visible and not heavily physically damaged
- Very low contrast or extreme blur may result in `readable: False`
- Low-confidence predictions (< 50%) are flagged — the raw image may be unreadable
