#!/usr/bin/env python3
"""Accuracy of the model on your own mouse drawings (saved with draw_digit.py).

Each saved drawing keeps its raw strokes, so this script re-runs the current
preprocessing and model on them: after changing preprocess.py or retraining,
run it again to see whether accuracy on your drawings got better.

Usage: python eval_my_digits.py
"""
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "model"))

from preprocess import strokes_to_mnist  # noqa: E402
from qmlp import QuantMLP  # noqa: E402


def main():
    files = sorted(glob.glob(os.path.join(HERE, "my_digits", "*.json")))
    if not files:
        sys.exit("No drawings yet. Run draw_digit.py, draw a digit and press its key (0-9) to save it.")
    model = QuantMLP.load(os.path.join(HERE, "..", "model", "quant_model.npz"))

    total = np.zeros(10, int)
    right = np.zeros(10, int)
    wrong = []
    for path in files:
        with open(path) as f:
            d = json.load(f)
        img = strokes_to_mnist(d["strokes"])
        if img is None:
            continue
        pred = int(model.predict(img)[0])
        total[d["label"]] += 1
        right[d["label"]] += pred == d["label"]
        if pred != d["label"]:
            wrong.append((os.path.basename(path), d["label"], pred))

    print(f"Accuracy on your drawings: {right.sum()}/{total.sum()} = {right.sum() / total.sum() * 100:.1f}%\n")
    print("digit  drawings  correct")
    for k in range(10):
        if total[k]:
            print(f"  {k}      {total[k]:4d}     {right[k]:4d}")
    if wrong:
        print("\nMisrecognized:")
        for name, label, pred in wrong:
            print(f"  {name}: drew {label}, model said {pred}")


if __name__ == "__main__":
    main()
