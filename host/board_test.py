#!/usr/bin/env python3
"""Run MNIST test images on the real board and check it against Python.

The same check as the Verilator simulation (sim/), but on the Basys 3: send
each image over the USB cable, compare the board's 10 logits with the Python
integer model (model/quant_model.npz), and report the clock cycles the board
measured plus the round-trip time measured on this computer.

Usage:
  python board_test.py               100 images, finds the board by itself
  python board_test.py -n 1000
  python board_test.py --port COM5   (Mac: /dev/cu.usbserial-...)
  python board_test.py --list        show serial ports

Needs: pip install numpy pyserial
"""
import argparse
import os
import sys
import urllib.request

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, os.path.join(ROOT, "model"))

from board import Board, BoardError, CLOCK_HZ, describe_ports  # noqa: E402
from qmlp import QuantMLP  # noqa: E402
from train_quantize import MNIST_FILES, MNIST_MIRRORS, md5, read_idx  # noqa: E402

TEST_FILES = ("t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz")


def load_test_set():
    """MNIST test images and labels: from firmware/test_data if present, else
    from model/data, downloading the two test files (2 MB) if needed."""
    td = os.path.join(ROOT, "firmware", "test_data")
    img_bin, lbl_bin = os.path.join(td, "test_images.bin"), os.path.join(td, "test_labels.bin")
    if os.path.exists(img_bin) and os.path.exists(lbl_bin):
        return np.fromfile(img_bin, np.uint8).reshape(-1, 784), np.fromfile(lbl_bin, np.uint8)

    data_dir = os.path.join(ROOT, "model", "data")
    os.makedirs(data_dir, exist_ok=True)
    for name in TEST_FILES:
        path = os.path.join(data_dir, name)
        for mirror in MNIST_MIRRORS:
            if os.path.exists(path) and md5(path) == MNIST_FILES[name]:
                break
            try:
                print(f"Downloading {name}")
                urllib.request.urlretrieve(mirror + name, path)
            except Exception as e:  # try the next mirror
                print(f"  failed: {e}")
        if not (os.path.exists(path) and md5(path) == MNIST_FILES[name]):
            sys.exit(f"Could not download {name} into {data_dir}.")
    images = read_idx(os.path.join(data_dir, TEST_FILES[0])).reshape(-1, 784)
    labels = read_idx(os.path.join(data_dir, TEST_FILES[1]))
    return images, labels


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-n", type=int, default=100, help="number of test images (default 100)")
    ap.add_argument("--port", help="serial port, e.g. COM5 or /dev/cu.usbserial-...")
    ap.add_argument("--list", action="store_true", help="list serial ports and exit")
    args = ap.parse_args()

    if args.list:
        print(describe_ports())
        return 0

    model_path = os.path.join(ROOT, "model", "quant_model.npz")
    if not os.path.exists(model_path):
        sys.exit("model/quant_model.npz not found. Run model/train_quantize.py first.")
    model = QuantMLP.load(model_path)
    images, labels = load_test_set()
    n = min(args.n, len(labels))
    ref = model.logits(images[:n])

    try:
        board = Board(args.port)
    except BoardError as e:
        sys.exit(str(e))

    with board:
        print(f"Board on {board.port} answered the ping. Running {n} MNIST test images...")
        mismatches, correct, cycles, trips = 0, 0, [], []
        for i in range(n):
            try:
                r = board.infer(images[i])
            except BoardError as e:
                sys.exit(f"\nImage {i}: {e}")
            same = np.array_equal(r.logits, ref[i])
            mismatches += not same
            correct += r.digit == labels[i]
            cycles.append(r.cycles)
            trips.append(r.round_trip_ms)
            if i < 5 or not same:
                print(f"  image {i}: label {labels[i]}, board says {r.digit}, {r.cycles} cycles, "
                      f"logits {'match' if same else 'DIFFER from'} Python")
            elif i % 20 == 0:
                print(f"  ... {i} / {n}")

    cycles = np.array(cycles)
    avg = cycles.mean()
    print()
    print(f"Logits identical to Python: {n - mismatches} / {n} images")
    print(f"Correct digits:             {correct} / {n}")
    print(f"Cycles per inference:       {avg:.0f} (min {cycles.min()}, max {cycles.max()})")
    print(f"Time per inference:         {avg / CLOCK_HZ * 1000:.2f} ms on the board (100 MHz)")
    print(f"Round trip per image:       {np.median(trips):.1f} ms measured on this computer "
          "(USB-UART both ways + inference)")
    if mismatches == n:
        print("FAIL: no image matches. Was hw/firmware.mem built from the same training run as "
              "model/quant_model.npz? Rebuild the firmware, copy it to hw/firmware.mem and rebuild "
              "the bitstream.")
    print("PASS: the board matches Python bit for bit" if mismatches == 0 else "FAIL")
    return 0 if mismatches == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
