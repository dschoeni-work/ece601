#!/usr/bin/env python3
"""Mouse drawing pad: draw a digit and see what the model sees and predicts.

The prediction uses the same integer model as the C firmware
(model/quant_model.npz), so the answer here is what the FPGA will give
for the same 28x28 image.

Controls
  drag with the left button   draw
  Space or Delete              clear
  Backspace or u               undo the last stroke
  0-9                          save the drawing as that digit to host/my_digits/
                               (to measure accuracy on your own drawings), then clear
  x                            delete the last saved drawing (pressed the wrong digit)

Usage: python draw_digit.py
"""
import datetime
import json
import os
import sys
import time
import tkinter as tk
from tkinter import ttk

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(HERE, "..", "model")
sys.path.insert(0, MODEL_DIR)

from preprocess import strokes_to_mnist  # noqa: E402
from qmlp import QuantMLP  # noqa: E402

SAVE_DIR = os.path.join(HERE, "my_digits")

CANVAS = 392            # drawing area, px (before Windows DPI scaling)
PEN_WIDTH = 18          # on-screen pen width, px (the model's pen width is set in preprocess.py)
PREVIEW_ZOOM = 6        # 28 x 6 = 168 px preview
BAR_W, BAR_H = 170, 18
PREDICT_EVERY = 0.05    # s, how often to re-run the model while drawing

BG, FG, MUTED = "#f5f5f7", "#1d1d1f", "#6e6e73"
INK, PAPER = "#ffffff", "#111111"
ACCENT, TRACK = "#2f6fde", "#dcdce1"


def load_model():
    path = os.path.join(MODEL_DIR, "quant_model.npz")
    if not os.path.exists(path):
        sys.exit("model/quant_model.npz not found. Run model/train_quantize.py first.")
    model = QuantMLP.load(path)
    # Integer logits times s_h * s_w2 are the real-valued logits, which give
    # meaningful softmax confidences.
    try:
        with open(os.path.join(MODEL_DIR, "quant_info.json")) as f:
            info = json.load(f)
        logit_scale = info["s_h"] * info["s_w2"]
    except (OSError, KeyError, ValueError):
        logit_scale = None
    return model, logit_scale


def softmax(z):
    e = np.exp(z - z.max())
    return e / e.sum()


class App:
    def __init__(self, root, model, logit_scale):
        self.root, self.model, self.logit_scale = root, model, logit_scale
        self.strokes = []
        self.saved = []         # drawings saved in this session, newest last
        self.pred = None
        self._last_predict = 0.0
        self._preview = None

        # On Windows, Tk is made DPI-aware in main(); scale pixel sizes to match.
        self.ui = root.winfo_fpixels("1i") / 96.0 if sys.platform == "win32" else 1.0
        px = lambda v: int(round(v * self.ui))  # noqa: E731
        self.px = px

        root.title("Mouse digit recognizer")
        root.configure(bg=BG)
        root.resizable(False, False)

        left = tk.Frame(root, bg=BG)
        left.grid(row=0, column=0, sticky="n", padx=(px(16), px(16)), pady=(px(16), 0))
        self.canvas = tk.Canvas(left, width=px(CANVAS), height=px(CANVAS), bg=PAPER,
                                highlightthickness=0, cursor="crosshair")
        self.canvas.pack()
        buttons = tk.Frame(left, bg=BG)
        buttons.pack(anchor="w", pady=(px(10), 0))
        ttk.Button(buttons, text="Clear (Space)", command=self.clear).pack(side="left")
        ttk.Button(buttons, text="Undo (Backspace)", command=self.undo).pack(side="left", padx=px(8))
        ttk.Button(buttons, text="Delete last save (x)", command=self.delete_last_save).pack(side="left")

        side = tk.Frame(root, bg=BG)
        side.grid(row=0, column=1, sticky="n", padx=(0, px(16)), pady=(px(16), 0))
        tk.Label(side, text="Model input (28 × 28)", bg=BG, fg=MUTED,
                 font=("Helvetica", 11)).pack(anchor="w")
        size = 28 * self._zoom()
        self.preview = tk.Label(side, bg=PAPER, width=size, height=size, bd=0)
        self.preview.pack(anchor="w", pady=(px(4), px(12)))

        tk.Label(side, text="Prediction", bg=BG, fg=MUTED, font=("Helvetica", 11)).pack(anchor="w")
        self.pred_label = tk.Label(side, text="–", bg=BG, fg=FG, font=("Helvetica", 48, "bold"))
        self.pred_label.pack(anchor="w")

        self.bars = tk.Canvas(side, width=px(BAR_W + 70), height=px(10 * (BAR_H + 4)),
                              bg=BG, highlightthickness=0)
        self.bars.pack(anchor="w", pady=(px(4), 0))

        self.status = tk.Label(root, bg=BG, fg=MUTED, font=("Helvetica", 11), anchor="w",
                               text="Draw a digit, then press 0–9 on the keyboard to save it with that label.")
        self.status.grid(row=1, column=0, columnspan=2, sticky="w", padx=px(16), pady=(px(10), px(14)))

        self.canvas.bind("<ButtonPress-1>", self.on_down)
        self.canvas.bind("<B1-Motion>", self.on_move)
        self.canvas.bind("<ButtonRelease-1>", self.on_up)
        for key in ("<space>", "<Delete>"):
            root.bind(key, lambda e: self.clear())
        for key in ("<BackSpace>", "u"):
            root.bind(key, lambda e: self.undo())
        for d in range(10):
            root.bind(str(d), lambda e, d=d: self.save(d))
        root.bind("x", lambda e: self.delete_last_save())

        self.show(None)

    def _zoom(self):
        return max(1, int(round(PREVIEW_ZOOM * self.ui)))

    # ---------------------------------------------------------- drawing

    def on_down(self, e):
        self.strokes.append([(e.x, e.y)])
        r = self.px(PEN_WIDTH) / 2
        self.canvas.create_oval(e.x - r, e.y - r, e.x + r, e.y + r, fill=INK, outline="",
                                tags=("ink", f"s{len(self.strokes) - 1}"))

    def on_move(self, e):
        if not self.strokes:
            return
        x0, y0 = self.strokes[-1][-1]
        if (e.x - x0) ** 2 + (e.y - y0) ** 2 < 4:  # ignore jitter under 2 px
            return
        self.strokes[-1].append((e.x, e.y))
        self.canvas.create_line(x0, y0, e.x, e.y, fill=INK, width=self.px(PEN_WIDTH),
                                capstyle="round", joinstyle="round",
                                tags=("ink", f"s{len(self.strokes) - 1}"))
        if time.monotonic() - self._last_predict > PREDICT_EVERY:
            self.predict()

    def on_up(self, e):
        self.predict()

    def clear(self):
        self.strokes = []
        self.canvas.delete("ink")
        self.predict()

    def undo(self):
        if self.strokes:
            self.canvas.delete(f"s{len(self.strokes) - 1}")
            self.strokes.pop()
            self.predict()

    # -------------------------------------------------------- inference

    def predict(self):
        self._last_predict = time.monotonic()
        img = strokes_to_mnist(self.strokes)
        if img is None:
            self.pred = None
            self.show(None)
            return
        logits = self.model.logits(img)[0]
        scale = self.logit_scale or 10.0 / max(1, int(np.abs(logits).max()))
        probs = softmax(logits * scale)
        self.pred = int(np.argmax(logits))
        self.show(img, probs)

    def show(self, img, probs=None):
        z = self._zoom()
        photo = tk.PhotoImage(width=28, height=28)
        if img is not None:
            rows = " ".join("{" + " ".join(f"#{v:02x}{v:02x}{v:02x}" for v in r) + "}" for r in img)
            photo.put(rows)
        self._preview = photo.zoom(z, z)
        self.preview.configure(image=self._preview, width=28 * z, height=28 * z)

        self.pred_label.configure(text="–" if self.pred is None else str(self.pred))
        b, px = self.bars, self.px
        b.delete("all")
        for k in range(10):
            y = px(k * (BAR_H + 4))
            p = 0.0 if probs is None else float(probs[k])
            top = probs is not None and k == self.pred
            b.create_text(px(6), y + px(BAR_H) / 2, text=str(k), anchor="w",
                          fill=FG if top else MUTED, font=("Helvetica", 11, "bold" if top else "normal"))
            b.create_rectangle(px(24), y, px(24 + BAR_W), y + px(BAR_H), fill=TRACK, outline="")
            if p > 0:
                b.create_rectangle(px(24), y, px(24) + max(1, p * px(BAR_W)), y + px(BAR_H),
                                   fill=ACCENT if top else "#9aa6bd", outline="")
            if probs is not None:
                b.create_text(px(30 + BAR_W), y + px(BAR_H) / 2, text=f"{p * 100:.0f}%", anchor="w",
                              fill=FG if top else MUTED, font=("Helvetica", 10))

    # ----------------------------------------------------------- saving

    def save(self, label):
        if not self.strokes:
            self.status.configure(text="Draw something first, then press its digit to save it.")
            return
        os.makedirs(SAVE_DIR, exist_ok=True)
        stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f")[:-3]
        path = os.path.join(SAVE_DIR, f"{label}_{stamp}.json")
        with open(path, "w") as f:
            json.dump({"label": label, "prediction": self.pred, "canvas_size": CANVAS,
                       "strokes": [[[int(x), int(y)] for x, y in s] for s in self.strokes]}, f)
        self.saved.append(path)
        verdict = "correct" if self.pred == label else f"model said {self.pred}"
        self.status.configure(text=f"Saved as {label} ({verdict}). {self._count()} saved. "
                                   "Wrong digit? Press x to delete it.")
        self.strokes = []
        self.canvas.delete("ink")
        self.pred = None
        self.show(None)

    def delete_last_save(self):
        if not self.saved:
            self.status.configure(text="Nothing saved in this session to delete.")
            return
        path = self.saved.pop()
        if os.path.exists(path):
            os.remove(path)
        self.status.configure(text=f"Deleted {os.path.basename(path)}. {self._count()} saved.")

    def _count(self):
        if not os.path.isdir(SAVE_DIR):
            return 0
        return len([f for f in os.listdir(SAVE_DIR) if f.endswith(".json")])


def main():
    if sys.platform == "win32":  # sharp text on high-DPI Windows screens
        try:
            import ctypes
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except Exception:
            pass
    model, logit_scale = load_model()
    root = tk.Tk()
    App(root, model, logit_scale)
    root.after(200, root.focus_force)
    root.mainloop()


if __name__ == "__main__":
    main()
