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

With --board, every finished stroke is also sent to the RISC-V CPU on the
Basys 3 (programmed with hw/program.tcl), and the window shows the board's
answer, its inference time and whether its logits match the laptop's.

Usage:
  python draw_digit.py                  laptop only
  python draw_digit.py --board          also run on the board (finds it by itself)
  python draw_digit.py --port COM5      same, with the serial port given
"""
import argparse
import datetime
import json
import os
import queue
import sys
import threading
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


class BoardLink:
    """Runs board.infer() on a background thread so the window never freezes.

    Only the newest image is kept: if the user draws faster than the board
    answers (about 0.1 s per image), older images are skipped.
    """

    def __init__(self, board):
        self.board = board
        self.results = queue.Queue()     # (request id, image, Result or Exception)
        self._cv = threading.Condition()
        self._job = None
        threading.Thread(target=self._run, daemon=True).start()

    def submit(self, job_id, img):
        with self._cv:
            self._job = (job_id, img)
            self._cv.notify()

    def _run(self):
        while True:
            with self._cv:
                while self._job is None:
                    self._cv.wait()
                job_id, img = self._job
                self._job = None
            try:
                self.results.put((job_id, img, self.board.infer(img)))
            except Exception as e:  # BoardError or a serial error: shown in the window
                self.results.put((job_id, img, e))


class App:
    def __init__(self, root, model, logit_scale, link=None, port=None):
        self.root, self.model, self.logit_scale = root, model, logit_scale
        self.link, self.job_id = link, 0
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

        if link is not None:
            board = tk.Frame(left, bg=BG)
            board.pack(anchor="w", fill="x", pady=(px(14), 0))
            tk.Label(board, text=f"On the board (RISC-V on Basys 3, {port})", bg=BG, fg=MUTED,
                     font=("Helvetica", 11)).pack(anchor="w")
            row = tk.Frame(board, bg=BG)
            row.pack(anchor="w", fill="x")
            self.board_digit = tk.Label(row, text="–", bg=BG, fg=FG, width=2, anchor="w",
                                        font=("Helvetica", 36, "bold"))
            self.board_digit.pack(side="left")
            self.board_info = tk.Label(row, text="Lift the mouse to send the drawing to the board.",
                                       bg=BG, fg=MUTED, font=("Helvetica", 11), justify="left", anchor="w")
            self.board_info.pack(side="left", padx=(px(8), 0))
            root.after(30, self.poll_board)

        side = tk.Frame(root, bg=BG)
        side.grid(row=0, column=1, sticky="n", padx=(0, px(16)), pady=(px(16), 0))
        tk.Label(side, text="Model input (28 × 28)", bg=BG, fg=MUTED,
                 font=("Helvetica", 11)).pack(anchor="w")
        size = 28 * self._zoom()
        self.preview = tk.Label(side, bg=PAPER, width=size, height=size, bd=0)
        self.preview.pack(anchor="w", pady=(px(4), px(12)))

        tk.Label(side, text="Prediction" if link is None else "On the laptop (Python)", bg=BG, fg=MUTED,
                 font=("Helvetica", 11)).pack(anchor="w")
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
        self.send_to_board()

    def clear(self):
        self.strokes = []
        self.canvas.delete("ink")
        self.predict()
        self.send_to_board()

    def undo(self):
        if self.strokes:
            self.canvas.delete(f"s{len(self.strokes) - 1}")
            self.strokes.pop()
            self.predict()
            self.send_to_board()

    # ------------------------------------------------------------ board

    def send_to_board(self):
        """Send the current drawing to the board; answers to older drawings are ignored."""
        if self.link is None:
            return
        self.job_id += 1
        img = strokes_to_mnist(self.strokes)
        if img is None:
            self.board_digit.configure(text="–")
            self.board_info.configure(text="Lift the mouse to send the drawing to the board.")
            return
        self.board_info.configure(text="Running on the board…")
        self.link.submit(self.job_id, img)

    def poll_board(self):
        try:
            while True:
                job_id, img, r = self.link.results.get_nowait()
                if job_id != self.job_id:
                    continue                      # the drawing changed since it was sent
                if isinstance(r, Exception):
                    self.board_digit.configure(text="?")
                    self.board_info.configure(text=f"Board error: {r}")
                    continue
                same = np.array_equal(r.logits, self.model.logits(img)[0])
                self.board_digit.configure(text=str(r.digit))
                self.board_info.configure(
                    text=f"{r.compute_ms:.2f} ms on the board ({r.cycles:,} cycles)\n"
                         f"{r.round_trip_ms:.0f} ms round trip, including the USB cable\n"
                         + ("Same logits as the laptop ✓" if same else "Logits DIFFER from the laptop"))
        except queue.Empty:
            pass
        self.root.after(30, self.poll_board)

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
        self.send_to_board()

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
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--board", action="store_true", help="also run each drawing on the Basys 3")
    ap.add_argument("--port", help="the board's serial port, e.g. COM5 (implies --board)")
    args = ap.parse_args()

    if sys.platform == "win32":  # sharp text on high-DPI Windows screens
        try:
            import ctypes
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except Exception:
            pass
    model, logit_scale = load_model()

    link = port = None
    if args.board or args.port:
        from board import Board, BoardError
        try:
            b = Board(args.port)
        except BoardError as e:
            sys.exit(str(e))
        link, port = BoardLink(b), b.port
        print(f"Connected to the board on {port}")

    root = tk.Tk()
    App(root, model, logit_scale, link, port)
    root.after(200, root.focus_force)
    root.mainloop()


if __name__ == "__main__":
    main()
