"""
Universal dot-matrix font editor.

Features:
  - Add / remove any character (any Unicode)
  - Any grid size per character: cols × rows set independently
  - Each character can have a different size
  - Save to font_custom.json after every action (auto-save)
  - Export to font_custom.py (ready for generate_sequences.py)

Usage:
    python font_editor.py
    python font_editor.py --font my_font.json   # open a specific file

Keyboard shortcuts:
  → / Enter   save current char and go to next
  ←           save current char and go to previous
  C           clear the grid
  Delete      remove current character (with confirmation)
"""

import argparse
import json
import tkinter as tk
from tkinter import messagebox, simpledialog
from pathlib import Path


# ── Defaults ────────────────────────────────────────────────────────────────

DEFAULT_COLS = 4
DEFAULT_ROWS = 6
MIN_COLS     = 1
MAX_COLS     = 20
MIN_ROWS     = 1
MAX_ROWS     = 20

MAX_CANVAS_W = 720   # px — canvas will not exceed this
MAX_CANVAS_H = 560
MIN_CELL     = 10    # px — minimum dot size
MAX_CELL     = 64    # px — maximum dot size
GAP          = 4     # px — gap between cells

DOT_ON  = "#1c1c1e"
DOT_OFF = "#ececec"
BORDER  = "#c0c0c0"

SAVE_FILE   = Path("font_custom.json")
EXPORT_FILE = Path("font_custom.py")


# ── Persistence ──────────────────────────────────────────────────────────────

def load_font(path: Path) -> tuple[dict, list]:
    """
    Load font from JSON file.
    Returns (font_data, chars_list) where:
      font_data   = {char: [[0,1,...], ...]}
      chars_list  = ordered list of all characters in the font
    """
    if path.exists():
        raw = json.loads(path.read_text(encoding="utf-8"))
        font_data  = {k: [list(r) for r in v] for k, v in raw.items()}
        chars_list = list(raw.keys())
        return font_data, chars_list

    # Seed from font5x7_cyrillic if available
    try:
        from font5x7_cyrillic import FONT
        font_data  = {}
        chars_list = []
        for ch, grid in FONT.items():
            rows_list = [list(map(int, row)) for row in grid.tolist()]
            font_data[ch]  = rows_list
            chars_list.append(ch)
        return font_data, chars_list
    except Exception:
        return {}, []


def save_font(font_data: dict, chars_list: list, path: Path):
    ordered = {ch: font_data[ch] for ch in chars_list if ch in font_data}
    path.write_text(
        json.dumps(ordered, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


# ── Cell-size helper ─────────────────────────────────────────────────────────

def compute_cell(cols: int, rows: int) -> int:
    cell_w = (MAX_CANVAS_W - (cols + 1) * GAP) // cols
    cell_h = (MAX_CANVAS_H - (rows + 1) * GAP) // rows
    return max(MIN_CELL, min(MAX_CELL, cell_w, cell_h))


# ── Add-character dialog ─────────────────────────────────────────────────────

class AddCharDialog(simpledialog.Dialog):
    def __init__(self, parent, existing: set):
        self.existing = existing
        self.result_char  = None
        self.result_cols  = DEFAULT_COLS
        self.result_rows  = DEFAULT_ROWS
        super().__init__(parent, title="Add character")

    def body(self, master):
        master.config(bg="#f5f5f7")

        tk.Label(master, text="Character:", bg="#f5f5f7",
                 font=("Segoe UI", 11)).grid(row=0, column=0, sticky="w", padx=8, pady=4)
        self.ent_char = tk.Entry(master, width=6, font=("Segoe UI", 16))
        self.ent_char.grid(row=0, column=1, sticky="w", padx=8)
        self.ent_char.focus_set()

        tk.Label(master, text="Columns:", bg="#f5f5f7",
                 font=("Segoe UI", 11)).grid(row=1, column=0, sticky="w", padx=8, pady=4)
        self.spn_cols = tk.Spinbox(master, from_=MIN_COLS, to=MAX_COLS,
                                   width=5, font=("Segoe UI", 11))
        self.spn_cols.delete(0, "end")
        self.spn_cols.insert(0, str(DEFAULT_COLS))
        self.spn_cols.grid(row=1, column=1, sticky="w", padx=8)

        tk.Label(master, text="Rows:", bg="#f5f5f7",
                 font=("Segoe UI", 11)).grid(row=2, column=0, sticky="w", padx=8, pady=4)
        self.spn_rows = tk.Spinbox(master, from_=MIN_ROWS, to=MAX_ROWS,
                                   width=5, font=("Segoe UI", 11))
        self.spn_rows.delete(0, "end")
        self.spn_rows.insert(0, str(DEFAULT_ROWS))
        self.spn_rows.grid(row=2, column=1, sticky="w", padx=8)

        return self.ent_char

    def validate(self):
        ch = self.ent_char.get().strip()
        if not ch:
            messagebox.showwarning("Invalid", "Enter a character.", parent=self)
            return False
        if len(ch) > 1:
            messagebox.showwarning("Invalid",
                                   "Enter exactly one character.", parent=self)
            return False
        if ch in self.existing:
            messagebox.showwarning("Duplicate",
                                   f"'{ch}' is already in the font.", parent=self)
            return False
        try:
            c = int(self.spn_cols.get())
            r = int(self.spn_rows.get())
        except ValueError:
            messagebox.showwarning("Invalid", "Cols/rows must be integers.", parent=self)
            return False
        if not (MIN_COLS <= c <= MAX_COLS) or not (MIN_ROWS <= r <= MAX_ROWS):
            messagebox.showwarning(
                "Invalid",
                f"Cols: {MIN_COLS}–{MAX_COLS}, Rows: {MIN_ROWS}–{MAX_ROWS}.",
                parent=self,
            )
            return False
        self.result_char = ch
        self.result_cols = c
        self.result_rows = r
        return True

    def apply(self):
        pass   # result_* already set in validate


# ── Main editor ──────────────────────────────────────────────────────────────

class FontEditor:

    def __init__(self, root: tk.Tk, save_path: Path):
        self.root      = root
        self.save_path = save_path

        self.root.title("Dot-Matrix Font Editor")
        self.root.configure(bg="#f5f5f7")
        self.root.resizable(True, True)

        self.font_data:  dict      = {}
        self.chars_list: list[str] = []
        self.idx:        int       = 0
        self.grid:       list[list[int]] = []
        self._cell_ids:  list[list[int]] = []

        self.font_data, self.chars_list = load_font(save_path)

        self._build_ui()
        if self.chars_list:
            self._load_char(0)
        else:
            self._update_empty_state()

    # ── Build UI ─────────────────────────────────────────────────────────

    def _build_ui(self):
        # Header
        hdr = tk.Frame(self.root, bg="#1c1c1e", padx=18, pady=10)
        hdr.pack(fill="x")

        self.lbl_char = tk.Label(hdr, text="—",
                                 font=("Segoe UI", 36, "bold"),
                                 fg="white", bg="#1c1c1e", width=4, anchor="w")
        self.lbl_char.pack(side="left")

        info = tk.Frame(hdr, bg="#1c1c1e")
        info.pack(side="left", padx=16)

        self.lbl_dim = tk.Label(info, text="",
                                font=("Segoe UI", 12), fg="#98c5e9", bg="#1c1c1e")
        self.lbl_dim.pack(anchor="w")

        self.lbl_progress = tk.Label(hdr, text="",
                                     font=("Segoe UI", 11), fg="#ccc", bg="#1c1c1e")
        self.lbl_progress.pack(side="right", padx=8)

        # Grid-size controls
        ctrl = tk.Frame(self.root, bg="#f5f5f7", pady=8)
        ctrl.pack()

        def _spinbox(parent, var, label, lo, hi, cmd):
            f = tk.Frame(parent, bg="#f5f5f7")
            f.pack(side="left", padx=10)
            tk.Label(f, text=label, font=("Segoe UI", 11),
                     bg="#f5f5f7").pack(side="left")
            spn = tk.Spinbox(f, textvariable=var,
                             from_=lo, to=hi, width=4,
                             font=("Segoe UI", 12), command=cmd)
            spn.pack(side="left", padx=4)
            spn.bind("<FocusOut>", lambda _: cmd())
            spn.bind("<Return>",   lambda _: cmd())
            return spn

        self.var_cols = tk.IntVar(value=DEFAULT_COLS)
        self.var_rows = tk.IntVar(value=DEFAULT_ROWS)

        _spinbox(ctrl, self.var_cols, "Columns:", MIN_COLS, MAX_COLS, self._on_cols_change)
        _spinbox(ctrl, self.var_rows, "Rows:",    MIN_ROWS, MAX_ROWS, self._on_rows_change)

        # Canvas (resizable)
        cvs_outer = tk.Frame(self.root, bg="#f5f5f7", pady=4)
        cvs_outer.pack()

        self.canvas = tk.Canvas(cvs_outer,
                                width=MAX_CANVAS_W, height=MAX_CANVAS_H,
                                bg="#f5f5f7", highlightthickness=0)
        self.canvas.pack()
        self.canvas.tag_bind("cell", "<Button-1>", self._on_click)

        # Navigation
        nav = tk.Frame(self.root, bg="#f5f5f7", pady=6)
        nav.pack()

        _b = dict(font=("Segoe UI", 12), relief="flat",
                  padx=14, pady=6, cursor="hand2", bd=0)

        self.btn_prev = tk.Button(nav, text="← Prev", command=self.prev_char,
                                  bg="#8e8e93", fg="white", **_b)
        self.btn_prev.pack(side="left", padx=5)

        tk.Button(nav, text="Clear", command=self.clear_grid,
                  bg="#ff3b30", fg="white", **_b).pack(side="left", padx=5)

        self.btn_next = tk.Button(nav, text="Save & Next →",
                                  command=self.save_and_next,
                                  bg="#34c759", fg="white", **_b)
        self.btn_next.pack(side="left", padx=5)

        # Add / Remove
        mgmt = tk.Frame(self.root, bg="#f5f5f7", pady=4)
        mgmt.pack()

        tk.Button(mgmt, text="+ Add character", command=self._add_char,
                  bg="#007aff", fg="white", **_b).pack(side="left", padx=5)

        tk.Button(mgmt, text="− Remove current", command=self._remove_char,
                  bg="#ff9500", fg="white", **_b).pack(side="left", padx=5)

        # Overview strip (scrollable canvas)
        sep = tk.Frame(self.root, bg="#d0d0d0", height=1)
        sep.pack(fill="x", pady=4)

        ov_outer = tk.Frame(self.root, bg="#f5f5f7", padx=10, pady=4)
        ov_outer.pack(fill="x")

        tk.Label(ov_outer, text="Characters  (● = defined, click to jump):",
                 font=("Segoe UI", 9), fg="#888", bg="#f5f5f7").pack(anchor="w")

        # Use a Frame with wraplength for the overview buttons
        self.ov_frame = tk.Frame(ov_outer, bg="#f5f5f7")
        self.ov_frame.pack(fill="x")
        self.ov_btns: dict[str, tk.Label] = {}
        self._build_overview()

        # Bottom bar
        sep2 = tk.Frame(self.root, bg="#d0d0d0", height=1)
        sep2.pack(fill="x")

        bot = tk.Frame(self.root, bg="#f5f5f7", padx=12, pady=8)
        bot.pack(fill="x")

        tk.Button(bot, text="Export → font_custom.py",
                  command=self.export_font,
                  font=("Segoe UI", 11), relief="flat",
                  bg="#5856d6", fg="white",
                  padx=12, pady=5, cursor="hand2").pack(side="left")

        self.lbl_status = tk.Label(bot, text="", font=("Segoe UI", 10),
                                   fg="#555", bg="#f5f5f7")
        self.lbl_status.pack(side="right")

        # Keyboard shortcuts
        self.root.bind("<Right>",  lambda _: self.save_and_next())
        self.root.bind("<Return>", lambda _: self.save_and_next())
        self.root.bind("<Left>",   lambda _: self.prev_char())
        self.root.bind("c",        lambda _: self.clear_grid())
        self.root.bind("C",        lambda _: self.clear_grid())
        self.root.bind("<Delete>", lambda _: self._remove_char())

    # ── Overview ─────────────────────────────────────────────────────────

    def _build_overview(self):
        for w in self.ov_frame.winfo_children():
            w.destroy()
        self.ov_btns.clear()

        PER_ROW = 20
        row_frame = None
        for i, ch in enumerate(self.chars_list):
            if i % PER_ROW == 0:
                row_frame = tk.Frame(self.ov_frame, bg="#f5f5f7")
                row_frame.pack(anchor="w")
            lbl = tk.Label(row_frame, text=ch,
                           font=("Segoe UI", 10), padx=3, pady=1,
                           bg="#f5f5f7", cursor="hand2")
            lbl.pack(side="left")
            lbl.bind("<Button-1>", lambda e, idx=i: self._jump_to(idx))
            self.ov_btns[ch] = lbl
        self._update_overview_colors()

    def _update_overview_colors(self):
        for i, ch in enumerate(self.chars_list):
            lbl = self.ov_btns.get(ch)
            if not lbl:
                continue
            current = (i == self.idx)
            defined = ch in self.font_data
            if current:
                lbl.config(fg="#ff9500", font=("Segoe UI", 10, "bold underline"))
            elif defined:
                lbl.config(fg="#34c759", font=("Segoe UI", 10, "bold"))
            else:
                lbl.config(fg="#aaaaaa", font=("Segoe UI", 10))

    # ── Canvas drawing ────────────────────────────────────────────────────

    def _draw_grid(self):
        cols = len(self.grid[0]) if self.grid else self.var_cols.get()
        rows = len(self.grid)    if self.grid else self.var_rows.get()

        cell = compute_cell(cols, rows)
        cw   = cols * cell + (cols + 1) * GAP
        ch   = rows * cell + (rows + 1) * GAP

        self.canvas.config(width=cw, height=ch)
        self.canvas.delete("all")
        self._cell_ids = []

        for r in range(rows):
            row_ids = []
            for c in range(cols):
                x1 = GAP + c * (cell + GAP)
                y1 = GAP + r * (cell + GAP)
                x2 = x1 + cell
                y2 = y1 + cell
                val = self.grid[r][c] if (r < len(self.grid) and
                                           c < len(self.grid[r])) else 0
                cid = self.canvas.create_rectangle(
                    x1, y1, x2, y2,
                    fill=DOT_ON if val else DOT_OFF,
                    outline=BORDER, width=1,
                    tags=("cell", f"r{r}c{c}"),
                )
                row_ids.append(cid)
            self._cell_ids.append(row_ids)

        self.root.update_idletasks()

    def _on_click(self, event):
        items = self.canvas.find_closest(event.x, event.y)
        if not items:
            return
        for tag in self.canvas.gettags(items[0]):
            if tag.startswith("r") and "c" in tag:
                try:
                    r_s, c_s = tag[1:].split("c")
                    r, c = int(r_s), int(c_s)
                    self.grid[r][c] ^= 1
                    self.canvas.itemconfig(
                        items[0],
                        fill=DOT_ON if self.grid[r][c] else DOT_OFF,
                    )
                except ValueError:
                    pass
                break

    # ── Grid resize helpers ───────────────────────────────────────────────

    def _resize_grid(self, new_cols: int, new_rows: int):
        old_rows = len(self.grid)
        old_cols = len(self.grid[0]) if self.grid else 0

        # Adjust columns
        new_grid = []
        for r in range(old_rows):
            row = self.grid[r]
            if new_cols > old_cols:
                row = row + [0] * (new_cols - old_cols)
            else:
                row = row[:new_cols]
            new_grid.append(row)

        # Adjust rows
        if new_rows > old_rows:
            for _ in range(new_rows - old_rows):
                new_grid.append([0] * new_cols)
        else:
            new_grid = new_grid[:new_rows]

        self.grid = new_grid

    def _on_cols_change(self):
        if not self.chars_list:
            return
        try:
            new_cols = int(self.var_cols.get())
        except (ValueError, tk.TclError):
            return
        new_cols = max(MIN_COLS, min(MAX_COLS, new_cols))
        self.var_cols.set(new_cols)
        new_rows = len(self.grid)
        self._resize_grid(new_cols, new_rows)
        self._draw_grid()
        self._update_header()

    def _on_rows_change(self):
        if not self.chars_list:
            return
        try:
            new_rows = int(self.var_rows.get())
        except (ValueError, tk.TclError):
            return
        new_rows = max(MIN_ROWS, min(MAX_ROWS, new_rows))
        self.var_rows.set(new_rows)
        new_cols = len(self.grid[0]) if self.grid else self.var_cols.get()
        self._resize_grid(new_cols, new_rows)
        self._draw_grid()
        self._update_header()

    # ── Navigation ────────────────────────────────────────────────────────

    def _load_char(self, idx: int):
        if not self.chars_list:
            return
        self.idx = idx
        ch = self.chars_list[idx]

        if ch in self.font_data:
            saved      = self.font_data[ch]
            saved_rows = len(saved)
            saved_cols = len(saved[0]) if saved else DEFAULT_COLS
            self.var_cols.set(saved_cols)
            self.var_rows.set(saved_rows)
            self.grid = [list(row) for row in saved]
        else:
            cols = self.var_cols.get()
            rows = self.var_rows.get()
            self.grid = [[0] * cols for _ in range(rows)]

        self._draw_grid()
        self._update_header()
        self._update_overview_colors()

    def _update_header(self):
        if not self.chars_list:
            self.lbl_char.config(text="—")
            self.lbl_dim.config(text="No characters")
            self.lbl_progress.config(text="0 / 0")
            return

        ch   = self.chars_list[self.idx]
        cols = len(self.grid[0]) if self.grid else self.var_cols.get()
        rows = len(self.grid)    if self.grid else self.var_rows.get()
        done = sum(1 for c in self.chars_list if c in self.font_data)

        self.lbl_char.config(text=ch)
        self.lbl_dim.config(text=f"{cols} cols × {rows} rows")
        self.lbl_progress.config(
            text=f"{self.idx + 1} / {len(self.chars_list)}   |   {done} defined")

        self.btn_prev.config(state="normal" if self.idx > 0 else "disabled")
        last = (self.idx == len(self.chars_list) - 1)
        self.btn_next.config(text="Save  ✓" if last else "Save & Next →")

    def _update_empty_state(self):
        self.lbl_char.config(text="—")
        self.lbl_dim.config(text="Font is empty — click '+ Add character'")
        self.lbl_progress.config(text="0 / 0")
        self.canvas.delete("all")

    def _save_current(self):
        if not self.chars_list:
            return
        ch = self.chars_list[self.idx]
        self.font_data[ch] = [list(row) for row in self.grid]
        save_font(self.font_data, self.chars_list, self.save_path)

    def save_and_next(self):
        if not self.chars_list:
            return
        self._save_current()
        if self.idx < len(self.chars_list) - 1:
            prev_ch = self.chars_list[self.idx]
            self._load_char(self.idx + 1)
            self.lbl_status.config(text=f"Saved  {prev_ch}")
        else:
            self._update_overview_colors()
            self._update_header()
            self.lbl_status.config(text="All characters saved!")
            messagebox.showinfo("Done",
                                "All characters defined!\n"
                                "Click 'Export' to generate font_custom.py.")

    def prev_char(self):
        if self.chars_list and self.idx > 0:
            self._save_current()
            self._load_char(self.idx - 1)

    def _jump_to(self, idx: int):
        if self.chars_list:
            self._save_current()
            self._load_char(idx)

    def clear_grid(self):
        if not self.grid:
            return
        cols = len(self.grid[0])
        rows = len(self.grid)
        self.grid = [[0] * cols for _ in range(rows)]
        self._draw_grid()

    # ── Add / Remove ──────────────────────────────────────────────────────

    def _add_char(self):
        dlg = AddCharDialog(self.root, existing=set(self.chars_list))
        if dlg.result_char is None:
            return

        ch   = dlg.result_char
        cols = dlg.result_cols
        rows = dlg.result_rows

        # Save current before switching
        if self.chars_list:
            self._save_current()

        self.chars_list.append(ch)
        self.font_data[ch] = [[0] * cols for _ in range(rows)]
        save_font(self.font_data, self.chars_list, self.save_path)

        self._build_overview()
        self.var_cols.set(cols)
        self.var_rows.set(rows)
        self._load_char(len(self.chars_list) - 1)
        self.lbl_status.config(text=f"Added  '{ch}'  ({cols}×{rows})")

    def _remove_char(self):
        if not self.chars_list:
            return
        ch = self.chars_list[self.idx]
        if not messagebox.askyesno(
                "Remove character",
                f"Remove '{ch}' from the font?\nThis cannot be undone."):
            return

        self.chars_list.pop(self.idx)
        self.font_data.pop(ch, None)
        save_font(self.font_data, self.chars_list, self.save_path)

        self._build_overview()

        if not self.chars_list:
            self.idx = 0
            self._update_empty_state()
        else:
            new_idx = min(self.idx, len(self.chars_list) - 1)
            self._load_char(new_idx)

        self.lbl_status.config(text=f"Removed  '{ch}'")

    # ── Export ────────────────────────────────────────────────────────────

    def export_font(self):
        if not self.font_data:
            messagebox.showwarning("No data", "No characters defined yet.")
            return

        # Save current char first
        if self.chars_list:
            self._save_current()

        lines = [
            '"""',
            "Custom dot-matrix font.",
            "Generated by font_editor.py — edit with font_editor.py",
            "",
            "Each entry: FONT[char] = numpy array of shape (rows, cols).",
            "Characters can have different sizes.",
            '"""',
            "import numpy as np",
            "",
            "",
            "FONT = {",
        ]

        for ch in self.chars_list:
            if ch not in self.font_data:
                continue
            grid  = self.font_data[ch]
            rows  = len(grid)
            cols  = len(grid[0]) if grid else 0
            rows_s = ", ".join(str(list(r)) for r in grid)
            lines.append(
                f"    {repr(ch)}: np.array([{rows_s}], dtype=np.uint8),"
                f"  # {cols}×{rows}"
            )

        lines += [
            "}",
            "",
            "FONT_5x7 = FONT   # backward-compat alias",
            "",
        ]

        EXPORT_FILE.write_text("\n".join(lines), encoding="utf-8")
        n = len([c for c in self.chars_list if c in self.font_data])
        self.lbl_status.config(text=f"Exported {n} chars → {EXPORT_FILE}")
        messagebox.showinfo(
            "Exported",
            f"Saved to  {EXPORT_FILE}\n{n} characters exported.",
        )


# ── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Dot-matrix font editor")
    ap.add_argument("--font", default=str(SAVE_FILE),
                    help=f"JSON font file to edit (default: {SAVE_FILE})")
    args = ap.parse_args()

    root = tk.Tk()
    FontEditor(root, save_path=Path(args.font))
    root.mainloop()
