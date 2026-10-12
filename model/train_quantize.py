#!/usr/bin/env python3
"""Phase 0: train an MNIST MLP in NumPy, quantize it to INT8, and export it
for the C/RISC-V firmware.

Usage:
    python train_quantize.py                 # 784-64-10, 20 epochs (~30 s)
    python train_quantize.py --epochs 30

Outputs (default: ../firmware):
    mlp_weights.h                INT8 weights, INT32 biases, requant constants
    test_data/test_images.bin    10000 x 784 uint8, row-major
    test_data/test_labels.bin    10000 uint8
    test_data/test_ref.bin       10000 x 10 int32 little-endian logits from the
                                 Python integer model (the C code must match)
and quant_model.npz next to this script, for the mouse canvas later.

Only NumPy is required.
"""
import argparse
import gzip
import hashlib
import json
import os
import sys
import time
import urllib.request

import numpy as np

from qmlp import QuantMLP, exact_matmul, input_q

HERE = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------- data

MNIST_FILES = {
    "train-images-idx3-ubyte.gz": "f68b3c2dcbeaaa9fbdd348bbdeb94873",
    "train-labels-idx1-ubyte.gz": "d53e105ee54ea40749a09fcbcd1e9432",
    "t10k-images-idx3-ubyte.gz": "9fb629c4189551a2d022fa330f9573f3",
    "t10k-labels-idx1-ubyte.gz": "ec29112dd5afa0611ce80d1b7f02629c",
}
MNIST_MIRRORS = [
    "https://ossci-datasets.s3.amazonaws.com/mnist/",
    "https://storage.googleapis.com/cvdf-datasets/mnist/",
    "https://raw.githubusercontent.com/fgnt/mnist/master/",
]


def md5(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def download_mnist(data_dir):
    os.makedirs(data_dir, exist_ok=True)
    for name, digest in MNIST_FILES.items():
        path = os.path.join(data_dir, name)
        if os.path.exists(path) and md5(path) == digest:
            continue
        for mirror in MNIST_MIRRORS:
            try:
                print(f"  downloading {name} from {mirror}")
                urllib.request.urlretrieve(mirror + name, path)
                if md5(path) == digest:
                    break
                print("    checksum mismatch, trying next mirror")
            except Exception as e:  # network error: try the next mirror
                print(f"    failed: {e}")
        else:
            sys.exit(f"Could not download {name}. Download it by hand into {data_dir}.")


def read_idx(path):
    with gzip.open(path, "rb") as f:
        data = f.read()
    ndim = data[3]
    shape = [int.from_bytes(data[4 + 4 * i: 8 + 4 * i], "big") for i in range(ndim)]
    return np.frombuffer(data, dtype=np.uint8, offset=4 + 4 * ndim).reshape(shape)


def load_mnist(data_dir):
    download_mnist(data_dir)
    rd = lambda n: read_idx(os.path.join(data_dir, n))
    x_tr = rd("train-images-idx3-ubyte.gz").reshape(-1, 784)
    y_tr = rd("train-labels-idx1-ubyte.gz")
    x_te = rd("t10k-images-idx3-ubyte.gz").reshape(-1, 784)
    y_te = rd("t10k-labels-idx1-ubyte.gz")
    return x_tr, y_tr, x_te, y_te


def to_float(img_u8):
    """Float input that equals the quantized input exactly: (pixel >> 1) / 127."""
    return input_q(img_u8).astype(np.float32) / 127.0


# ------------------------------------------------------------ training

def forward(params, x):
    w1, b1, w2, b2 = params
    z1 = x @ w1.T + b1
    h = np.maximum(z1, 0)
    return z1, h, h @ w2.T + b2


def float_accuracy(params, x_u8, y):
    correct = 0
    for i in range(0, len(y), 2000):
        _, _, z2 = forward(params, to_float(x_u8[i:i + 2000]))
        correct += int((z2.argmax(1) == y[i:i + 2000]).sum())
    return correct / len(y)


def train(x_tr, y_tr, x_te, y_te, hidden, epochs, batch, lr, seed):
    rng = np.random.default_rng(seed)
    params = [
        (rng.standard_normal((hidden, 784)) * np.sqrt(2 / 784)).astype(np.float32),
        np.zeros(hidden, np.float32),
        (rng.standard_normal((10, hidden)) * np.sqrt(2 / hidden)).astype(np.float32),
        np.zeros(10, np.float32),
    ]
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    b1_, b2_, eps, t = 0.9, 0.999, 1e-8, 0
    x_all = to_float(x_tr)
    n = len(y_tr)

    for epoch in range(epochs):
        step_lr = lr * 0.5 * (1 + np.cos(np.pi * epoch / epochs))  # cosine decay
        perm = rng.permutation(n)
        t0 = time.time()
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            xb, yb = x_all[idx], y_tr[idx]
            z1, h, z2 = forward(params, xb)
            p = np.exp(z2 - z2.max(1, keepdims=True))
            p /= p.sum(1, keepdims=True)
            p[np.arange(len(yb)), yb] -= 1
            g_z2 = p / len(yb)
            g_z1 = (g_z2 @ params[2]) * (z1 > 0)
            grads = [g_z1.T @ xb, g_z1.sum(0), g_z2.T @ h, g_z2.sum(0)]
            t += 1
            for k, g in enumerate(grads):  # Adam
                m[k] = b1_ * m[k] + (1 - b1_) * g
                v[k] = b2_ * v[k] + (1 - b2_) * g * g
                mh = m[k] / (1 - b1_ ** t)
                vh = v[k] / (1 - b2_ ** t)
                params[k] -= (step_lr * mh / (np.sqrt(vh) + eps)).astype(np.float32)
        acc = float_accuracy(params, x_te, y_te)
        print(f"  epoch {epoch + 1:2d}/{epochs}  test acc {acc * 100:.2f}%  ({time.time() - t0:.1f}s)")
    return params


# -------------------------------------------------------- quantization

def quantize_multiplier(real):
    """real (> 0) ~= m * 2**-shift with m in [2**30, 2**31)."""
    e = int(np.floor(np.log2(real)))
    shift = 30 - e
    m = int(round(real * 2.0 ** shift))
    if m == 2 ** 31:
        m //= 2
        shift -= 1
    assert 2 ** 30 <= m < 2 ** 31 and 1 <= shift <= 62, (m, shift)
    return m, shift


def quantize(params, x_calib_u8, percentile):
    w1, b1, w2, b2 = [p.astype(np.float64) for p in params]
    s_x = 1 / 127

    s_w1 = np.abs(w1).max() / 127
    w1_q = np.clip(np.rint(w1 / s_w1), -127, 127).astype(np.int8)
    b1_q = np.rint(b1 / (s_x * s_w1)).astype(np.int64)

    # Hidden-activation scale from layer-1 outputs on calibration images.
    acc1 = exact_matmul(input_q(x_calib_u8), w1_q) + b1_q
    h_real = np.maximum(acc1, 0) * (s_x * s_w1)
    s_h = np.percentile(h_real, percentile) / 127
    m1, shift1 = quantize_multiplier(s_x * s_w1 / s_h)

    s_w2 = np.abs(w2).max() / 127
    w2_q = np.clip(np.rint(w2 / s_w2), -127, 127).astype(np.int8)
    b2_q = np.rint(b2 / (s_h * s_w2)).astype(np.int64)

    scales = {"s_x": s_x, "s_w1": s_w1, "s_h": s_h, "s_w2": s_w2,
              "m1": m1, "shift1": shift1, "calib_percentile": percentile}
    return QuantMLP(w1_q, b1_q, m1, shift1, w2_q, b2_q), scales


# -------------------------------------------------------------- export

def c_array(ctype, name, size_expr, values, per_line):
    lines = []
    for i in range(0, len(values), per_line):
        lines.append("    " + ", ".join(str(int(v)) for v in values[i:i + per_line]) + ",")
    return f"static const {ctype} {name}[{size_expr}] = {{\n" + "\n".join(lines) + "\n};\n"


def write_header(q, path, info):
    hidden = q.hidden
    with open(path, "w", newline="\n") as f:
        f.write("/* Generated by model/train_quantize.py -- do not edit by hand.\n")
        f.write(f" * MNIST MLP 784-{hidden}-10, INT8 weights, integer-only inference.\n")
        f.write(f" * Python integer model test accuracy: {info['int_acc'] * 100:.2f}%\n */\n")
        f.write("#ifndef MLP_WEIGHTS_H\n#define MLP_WEIGHTS_H\n\n#include <stdint.h>\n\n")
        f.write(f"#define MLP_WEIGHTS_IN     784\n#define MLP_WEIGHTS_HIDDEN {hidden}\n")
        f.write("#define MLP_WEIGHTS_OUT    10\n\n")
        f.write("/* Layer-1 requantization: h = clamp((acc * M1 + 2^(SHIFT1-1)) >> SHIFT1, 0, 127) */\n")
        f.write(f"#define MLP_M1     {q.m1}LL\n#define MLP_SHIFT1 {q.shift1}\n\n")
        f.write("/* W1: row j = weights of hidden unit j over the 784 pixels (28 per line = one image row) */\n")
        f.write(c_array("int8_t", "MLP_W1", f"{hidden} * 784", q.w1.ravel(), 28) + "\n")
        f.write(c_array("int32_t", "MLP_B1", str(hidden), q.b1, 8) + "\n")
        f.write(c_array("int8_t", "MLP_W2", f"10 * {hidden}", q.w2.ravel(), 16) + "\n")
        f.write(c_array("int32_t", "MLP_B2", "10", q.b2, 10) + "\n")
        f.write("#endif /* MLP_WEIGHTS_H */\n")


def write_test_data(q, x_te, y_te, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    logits = q.logits(x_te)
    assert np.abs(logits).max() < 2 ** 31
    x_te.astype(np.uint8).tofile(os.path.join(out_dir, "test_images.bin"))
    y_te.astype(np.uint8).tofile(os.path.join(out_dir, "test_labels.bin"))
    logits.astype("<i4").tofile(os.path.join(out_dir, "test_ref.bin"))


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--calib-percentile", type=float, default=99.99,
                    help="percentile of hidden activations mapped to 127")
    ap.add_argument("--data-dir", default=os.path.join(HERE, "data"))
    ap.add_argument("--out-dir", default=os.path.join(HERE, "..", "firmware"))
    args = ap.parse_args()

    print("Loading MNIST")
    x_tr, y_tr, x_te, y_te = load_mnist(args.data_dir)
    print(f"  train {x_tr.shape}, test {x_te.shape}")

    print(f"Training float MLP 784-{args.hidden}-10")
    params = train(x_tr, y_tr, x_te, y_te, args.hidden, args.epochs, args.batch, args.lr, args.seed)
    float_acc = float_accuracy(params, x_te, y_te)

    print("Quantizing to INT8")
    calib = x_tr[np.random.default_rng(args.seed + 1).permutation(len(x_tr))[:10000]]
    q, scales = quantize(params, calib, args.calib_percentile)
    _, acc1 = q.hidden_q(x_te)
    assert np.abs(acc1).max() < 2 ** 31, "layer-1 accumulator overflows int32"
    int_acc = float((q.predict(x_te) == y_te).mean())
    print(f"  float accuracy {float_acc * 100:.2f}%   int8 accuracy {int_acc * 100:.2f}%")
    print(f"  M1 = {q.m1} * 2^-{q.shift1}")

    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    info = {"float_acc": float_acc, "int_acc": int_acc, "hidden": args.hidden,
            "epochs": args.epochs, "seed": args.seed, **scales}
    write_header(q, os.path.join(out_dir, "mlp_weights.h"), info)
    write_test_data(q, x_te, y_te, os.path.join(out_dir, "test_data"))
    q.save(os.path.join(HERE, "quant_model.npz"))
    with open(os.path.join(HERE, "quant_info.json"), "w") as f:
        json.dump(info, f, indent=2)

    print(f"Wrote {os.path.join(out_dir, 'mlp_weights.h')}")
    print(f"Wrote {os.path.join(out_dir, 'test_data')}{os.sep}test_*.bin")
    print("Next: build and run firmware/test_mlp to check C matches Python.")


if __name__ == "__main__":
    main()
