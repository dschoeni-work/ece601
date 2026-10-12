#!/usr/bin/env python3
"""Checks for preprocess.py (no GUI needed).

1. Round trip: take MNIST test digits, blow them up to a random size and place
   them at a random spot on a 400x400 canvas (like a mouse drawing), run
   to_mnist(), and check the model is still about as accurate as on the
   original images. This catches bugs in cropping, scaling and centering.
2. Pen width: synthetic stroke drawings should have about as much ink as
   MNIST digits, and drawing the same strokes small or large must give the
   same image.

Usage: python test_preprocess.py
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "model"))

from preprocess import _shift, area_resize, strokes_to_mnist, to_mnist  # noqa: E402
from qmlp import QuantMLP  # noqa: E402
import train_quantize  # noqa: E402


def round_trip(model, x, y, rng, n):
    idx = rng.permutation(len(y))[:n]
    preds = []
    for i in idx:
        size = int(rng.integers(60, 300))
        big = area_resize(x[i].reshape(28, 28) / 255.0, size, size)
        canvas = np.zeros((400, 400))
        oy, ox = rng.integers(0, 400 - size, size=2)
        canvas[oy:oy + size, ox:ox + size] = big
        preds.append(model.predict(to_mnist(canvas))[0])
    acc_new = float((np.array(preds) == y[idx]).mean())
    acc_orig = float((model.predict(x[idx]) == y[idx]).mean())
    return acc_orig, acc_new


def ellipse(cx, cy, rx, ry, a0=0, a1=360, n=40):
    t = np.radians(np.linspace(a0, a1, n))
    return list(zip(cx + rx * np.cos(t), cy + ry * np.sin(t)))


# Simple drawings in a 0..100 box, y pointing down (screen coordinates).
SYNTHETIC = {
    0: [ellipse(50, 50, 28, 42)],
    1: [[(52, 8), (50, 92)]],
    2: [ellipse(50, 30, 26, 22, 200, 380) + [(24, 92), (78, 92)]],
    3: [ellipse(48, 28, 26, 20, 210, 450) + ellipse(48, 70, 28, 22, 270, 510)],
    4: [[(62, 8), (18, 64), (84, 64)], [(62, 30), (62, 94)]],
    5: [[(76, 10), (32, 10), (28, 46)] + ellipse(48, 66, 28, 26, 220, 500)],
    6: [[(66, 6), (36, 40)] + ellipse(50, 68, 26, 24, 200, 560)],
    7: [[(18, 12), (82, 12), (40, 94)]],
    8: [ellipse(50, 28, 22, 20) + ellipse(50, 70, 27, 23, -90, 270)],
    9: [ellipse(50, 32, 26, 24, 0, 360) + [(76, 32), (70, 94)]],
}


def main():
    model = QuantMLP.load(os.path.join(HERE, "..", "model", "quant_model.npz"))
    _, _, x_te, y_te = train_quantize.load_mnist(os.path.join(HERE, "..", "model", "data"))
    rng = np.random.default_rng(0)
    ok = True

    acc_orig, acc_new = round_trip(model, x_te, y_te, rng, 2000)
    print(f"Round trip on 2000 MNIST test digits: original {acc_orig * 100:.2f}%, "
          f"after preprocessing {acc_new * 100:.2f}%")
    if acc_new < acc_orig - 0.01:
        print("  FAIL: preprocessing loses more than 1 point of accuracy")
        ok = False

    nz = (x_te > 0).sum(axis=1).mean()
    dark = (x_te > 128).sum(axis=1).mean()
    print(f"MNIST ink per digit: {nz:.0f} px > 0, {dark:.0f} px > 128")
    correct = 0
    for label, strokes in SYNTHETIC.items():
        img = strokes_to_mnist(strokes)
        small = strokes_to_mnist([[(x * 0.3 + 5, y * 0.3 + 7) for x, y in s] for s in strokes])
        # Centering rounds to whole pixels, so a centre of mass sitting exactly on
        # a half pixel may land one pixel either way; allow that, nothing more.
        diff = min(np.abs(img.astype(int) - _shift(small, dy, dx).astype(int)).max()
                   for dy in (-1, 0, 1) for dx in (-1, 0, 1))
        if diff > 2:
            print(f"  FAIL: digit {label} changes with drawing size (max diff {diff})")
            ok = False
        pred = int(model.predict(img)[0])
        correct += pred == label
        print(f"  synthetic {label}: predicted {pred}, ink {(img > 0).sum()} px > 0, "
              f"{(img > 128).sum()} px > 128")
    print(f"Synthetic drawings: {correct}/10 recognized")

    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
