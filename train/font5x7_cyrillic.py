"""
Variable-width dot-matrix bitmap font for Ukrainian passport characters.

Actual grid dimensions measured from physical passports:
  - Standard characters:  4 columns x 6 rows
  - Digit '1':            3 columns x 6 rows
  - Letters M, T:         5 columns x 6 rows

Format: each character is a numpy array of shape (6, W), W in {3, 4, 5}.
  1 = dot present, 0 = empty.

Row encoding: each row value is an integer; bit (W-1-c) drives column c.
  Example (4-wide): 0x9 = 0b1001 = X..X  (col 0 and col 3 lit)

K vs H relationship (as measured on physical passports):
  H (Н) = standard H shape with 3-dot crossbar
  K (К) = Н + 1 extra dot at (row 1, col 3):
           one row above the crossbar (row 2), rightmost column
"""

import numpy as np


def _rows_to_grid(rows, width: int) -> np.ndarray:
    """Convert list of 6 row-values to a (6, width) numpy grid."""
    grid = np.zeros((6, width), dtype=np.uint8)
    for r, val in enumerate(rows[:6]):
        for c in range(width):
            grid[r, c] = (val >> (width - 1 - c)) & 1
    return grid


def _r4(rows): return _rows_to_grid(rows, 4)
def _r3(rows): return _rows_to_grid(rows, 3)
def _r5(rows): return _rows_to_grid(rows, 5)


# ── Digits ────────────────────────────────────────────────────────────
# 4x6, except '1' which is 3x6

DIGITS = {
    '0': _r4([0x6, 0x9, 0x9, 0x9, 0x9, 0x6]),   # .XX. / X..X / ... / .XX.
    '1': _r3([0x2, 0x6, 0x2, 0x2, 0x2, 0x7]),   # .X. / XX. / .X. / .X. / .X. / XXX
    '2': _r4([0x6, 0x1, 0x6, 0x8, 0x8, 0xF]),   # .XX. / ...X / .XX. / X... / X... / XXXX
    '3': _r4([0x6, 0x1, 0x6, 0x1, 0x1, 0x6]),   # .XX. / ...X / .XX. / ...X / ...X / .XX.
    '4': _r4([0x9, 0x9, 0xF, 0x1, 0x1, 0x1]),   # X..X / X..X / XXXX / ...X / ...X / ...X
    '5': _r4([0xF, 0x8, 0xE, 0x1, 0x1, 0xE]),   # XXXX / X... / XXX. / ...X / ...X / XXX.
    '6': _r4([0x6, 0x8, 0xE, 0x9, 0x9, 0x6]),   # .XX. / X... / XXX. / X..X / X..X / .XX.
    '7': _r4([0xF, 0x1, 0x2, 0x2, 0x2, 0x2]),   # XXXX / ...X / ..X. / ..X. / ..X. / ..X.
    '8': _r4([0x6, 0x9, 0x6, 0x9, 0x9, 0x6]),   # .XX. / X..X / .XX. / X..X / X..X / .XX.
    '9': _r4([0x6, 0x9, 0x9, 0x7, 0x1, 0x6]),   # .XX. / X..X / X..X / .XXX / ...X / .XX.
}

# ── Cyrillic uppercase ────────────────────────────────────────────────
# 4x6, except М and Т which are 5x6

CYRILLIC = {
    'А': _r4([0x6, 0x9, 0x9, 0xF, 0x9, 0x9]),   # .XX. / X..X / X..X / XXXX / X..X / X..X
    'Б': _r4([0xF, 0x8, 0xE, 0x9, 0x9, 0xE]),   # XXXX / X... / XXX. / X..X / X..X / XXX.
    'В': _r4([0xE, 0x9, 0xE, 0x9, 0x9, 0xE]),   # XXX. / X..X / XXX. / X..X / X..X / XXX.
    'Г': _r4([0xF, 0x8, 0x8, 0x8, 0x8, 0x8]),   # XXXX / X... / X... / X... / X... / X...
    'Ґ': _r4([0xF, 0x1, 0x8, 0x8, 0x8, 0x8]),   # XXXX / ...X(tick) / X... / X... / X... / X...
    'Д': _r4([0x6, 0xA, 0xA, 0xA, 0xF, 0x5]),   # .XX. / X.X. / X.X. / X.X. / XXXX / .X.X(feet)
    'Е': _r4([0xF, 0x8, 0xE, 0x8, 0x8, 0xF]),   # XXXX / X... / XXX. / X... / X... / XXXX
    'Є': _r4([0x7, 0x8, 0xE, 0x8, 0x8, 0x7]),   # .XXX / X... / XXX. / X... / X... / .XXX
    'Ж': _r4([0xA, 0xA, 0xF, 0xA, 0xA, 0xA]),   # X.X. / X.X. / XXXX / X.X. / X.X. / X.X.
    'З': _r4([0xE, 0x1, 0x6, 0x1, 0x1, 0xE]),   # XXX. / ...X / .XX. / ...X / ...X / XXX.
    'И': _r4([0x9, 0x9, 0xD, 0xB, 0x9, 0x9]),   # X..X / X..X / XX.X / X.XX / X..X / X..X
    'І': _r4([0x6, 0x2, 0x2, 0x2, 0x2, 0x6]),   # .XX. / ..X. / ..X. / ..X. / ..X. / .XX.
    'Ї': _r4([0xA, 0x6, 0x2, 0x2, 0x2, 0x6]),   # X.X.(dots) / .XX. / ..X. / ..X. / ..X. / .XX.
    'Й': _r4([0x5, 0x9, 0xD, 0xB, 0x9, 0x9]),   # .X.X(breve) / X..X / XX.X / X.XX / X..X / X..X
    'К': _r4([0xA, 0xB, 0xE, 0xA, 0xA, 0xA]),   # Н + dot at (row1, col3): X.X. / X.XX / XXX. / X.X. / X.X. / X.X.
    'Л': _r4([0x7, 0x2, 0x2, 0x2, 0x9, 0x9]),   # .XXX / ..X. / ..X. / ..X. / X..X / X..X
    'М': _r5([0x11, 0x1B, 0x15, 0x11, 0x11, 0x11]),  # X...X / XX.XX / X.X.X / X...X / X...X / X...X
    'Н': _r4([0xA, 0xA, 0xE, 0xA, 0xA, 0xA]),   # X.X. / X.X. / XXX. / X.X. / X.X. / X.X.
    'О': _r4([0x6, 0x9, 0x9, 0x9, 0x9, 0x6]),   # .XX. / X..X / X..X / X..X / X..X / .XX.
    'П': _r4([0xF, 0x9, 0x9, 0x9, 0x9, 0x9]),   # XXXX / X..X / X..X / X..X / X..X / X..X
    'Р': _r4([0xE, 0x9, 0xE, 0x8, 0x8, 0x8]),   # XXX. / X..X / XXX. / X... / X... / X...
    'С': _r4([0x7, 0x8, 0x8, 0x8, 0x8, 0x7]),   # .XXX / X... / X... / X... / X... / .XXX
    'Т': _r5([0x1F, 0x04, 0x04, 0x04, 0x04, 0x04]),  # XXXXX / ..X.. / ..X.. / ..X.. / ..X.. / ..X..
    'У': _r4([0x9, 0x9, 0x9, 0x6, 0x2, 0x2]),   # X..X / X..X / X..X / .XX. / ..X. / ..X.
    'Ф': _r4([0x4, 0xE, 0xA, 0xA, 0xE, 0x4]),   # ..X. / XXX. wait... 0x4=0100=.X.. 0xE=1110=XXX.
    'Х': _r4([0x9, 0x9, 0x6, 0x6, 0x9, 0x9]),   # X..X / X..X / .XX. / .XX. / X..X / X..X
    'Ц': _r4([0x9, 0x9, 0x9, 0x9, 0xF, 0x1]),   # X..X / X..X / X..X / X..X / XXXX / ...X(tail)
    'Ч': _r4([0x9, 0x9, 0x9, 0x7, 0x1, 0x1]),   # X..X / X..X / X..X / .XXX / ...X / ...X
    'Ш': _r4([0xB, 0xB, 0xB, 0xB, 0xB, 0xF]),   # X.XX / X.XX / X.XX / X.XX / X.XX / XXXX
    'Щ': _r4([0xB, 0xB, 0xB, 0xB, 0xF, 0x1]),   # X.XX / X.XX / X.XX / X.XX / XXXX / ...X(tail)
    'Ь': _r4([0x8, 0x8, 0xE, 0x9, 0x9, 0xE]),   # X... / X... / XXX. / X..X / X..X / XXX.
    'Ю': _r4([0xB, 0x9, 0xF, 0x9, 0x9, 0xB]),   # X.XX / X..X / XXXX / X..X / X..X / X.XX
    'Я': _r4([0xE, 0x9, 0xE, 0xA, 0x9, 0x9]),   # XXX. / X..X / XXX. / X.X. / X..X / X..X
}

# ── Combined font dict ────────────────────────────────────────────────

FONT: dict[str, np.ndarray] = {}
for _k, _v in DIGITS.items():
    FONT[_k] = _v
for _k, _v in CYRILLIC.items():
    FONT[_k] = _v

# Backward-compat alias (old name used shape 7x5, now 6x4/5/3)
FONT_5x7 = FONT


# ── Verification ──────────────────────────────────────────────────────

def print_char(char: str):
    grid = FONT.get(char)
    if grid is None:
        print(f"'{char}' not found")
        return
    rows, cols = grid.shape
    print(f"  {char}  ({cols}x{rows}):")
    for row in grid:
        print("  " + "".join("X" if v else "." for v in row))
    print()


def save_preview(path: str = "font_preview.png"):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    digits  = list("0123456789")
    cyrilic = list("АБВГҐДЕЄЖЗИІЇЙКЛМНОПРСТУФХЦЧШЩЬЮЯ")
    chars   = digits + cyrilic

    ncols = 10
    nrows = (len(chars) + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(ncols * 1.6, nrows * 2.4),
                              squeeze=False)
    axes_flat = axes.flatten()

    for idx, char in enumerate(chars):
        ax = axes_flat[idx]
        grid = FONT[char]
        ax.imshow(grid, cmap='binary', vmin=0, vmax=1,
                  interpolation='nearest', aspect='auto')
        ax.set_title(char, fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
        h, w = grid.shape
        for x in range(w + 1):
            ax.axvline(x - 0.5, color='lightgray', lw=0.4)
        for y in range(h + 1):
            ax.axhline(y - 0.5, color='lightgray', lw=0.4)

    for idx in range(len(chars), len(axes_flat)):
        axes_flat[idx].axis('off')

    plt.suptitle(f'Passport dot-matrix font  |  {len(chars)} chars  '
                 f'(4x6 standard, 3x6 for "1", 5x6 for M/T)',
                 fontsize=11)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


if __name__ == "__main__":
    save_preview()
    print("\nSample characters:")
    for c in ['0', '1', 'А', 'К', 'Н', 'М', 'Т', 'Ю', 'Я']:
        print_char(c)
    print(f"Total chars: {len(FONT)}")
    print("\nGrid sizes:")
    for ch, g in FONT.items():
        if g.shape != (6, 4):
            print(f"  {ch}: {g.shape[1]}x{g.shape[0]}")
