"""Integer-only MLP inference that matches firmware/mlp.c bit for bit.

The network is 784 -> HIDDEN -> 10:

    x_q   = pixel >> 1                                  (0..127, scale 1/127)
    acc1  = x_q . W1_q + B1_q                           (int32)
    h_q   = clamp((acc1 * M1 + 2^(SHIFT1-1)) >> SHIFT1, 0, 127)
    acc2  = h_q . W2_q + B2_q                           (int32 logits)
    digit = argmax(acc2)                                (first max wins)

The clamp at 0 is the ReLU. Every step is integer arithmetic, so the C code
and this file produce identical logits.
"""
import numpy as np


def input_q(img_u8):
    """28x28 uint8 image(s), flattened to (..., 784), to int64 values 0..127."""
    return (np.asarray(img_u8, dtype=np.uint8) >> 1).astype(np.int64)


def exact_matmul(a, w):
    """(N, K) int64 activations times (M, K) int8 weights -> (N, M) int64.

    The products and their sums are integers far below 2**53, so float64
    BLAS computes them exactly and much faster than an integer matmul.
    """
    return np.rint(a.astype(np.float64) @ w.astype(np.float64).T).astype(np.int64)


class QuantMLP:
    def __init__(self, w1, b1, m1, shift1, w2, b2):
        self.w1 = np.asarray(w1, dtype=np.int8)
        self.b1 = np.asarray(b1, dtype=np.int64)
        self.m1 = int(m1)
        self.shift1 = int(shift1)
        self.w2 = np.asarray(w2, dtype=np.int8)
        self.b2 = np.asarray(b2, dtype=np.int64)

    @property
    def hidden(self):
        return self.w1.shape[0]

    def hidden_q(self, img_u8):
        """Layer-1 output h_q (int64, 0..127) and the raw accumulator acc1."""
        acc1 = exact_matmul(input_q(img_u8), self.w1) + self.b1
        h = (acc1 * self.m1 + (1 << (self.shift1 - 1))) >> self.shift1
        return np.clip(h, 0, 127), acc1

    def logits(self, img_u8):
        """int64 logits, shape (N, 10). Accepts uint8 images shaped (784,),
        (28, 28), (N, 784) or (N, 28, 28)."""
        img = np.asarray(img_u8, dtype=np.uint8).reshape(-1, 784)
        h, _ = self.hidden_q(img)
        return exact_matmul(h, self.w2) + self.b2

    def predict(self, img_u8):
        return np.argmax(self.logits(img_u8), axis=1)

    def save(self, path):
        np.savez(path, w1=self.w1, b1=self.b1, m1=self.m1, shift1=self.shift1,
                 w2=self.w2, b2=self.b2)

    @classmethod
    def load(cls, path):
        d = np.load(path)
        return cls(d["w1"], d["b1"], d["m1"], d["shift1"], d["w2"], d["b2"])
