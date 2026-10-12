"""Turn mouse strokes into a 28x28 MNIST-style image.

MNIST digits were prepared like this (LeCun et al.):
  1. scale the digit so its longer side is 20 px, keeping the aspect ratio,
     with anti-aliasing (gray edges);
  2. paste it into a 28x28 field and shift it so its center of mass sits at
     pixel (14, 14).
A raw mouse drawing is off-center, any size, and has a fixed pen width, so we
redo the same steps here. The pen width is set relative to the drawing size,
so a small and a large drawing of the same digit give the same image.

Output: uint8 28x28, 0 = background, 255 = ink (same format as MNIST and as
the firmware input).
"""
import numpy as np

DIGIT_BOX = 20          # longer side of the digit inside the 28x28 field
FIELD = 28
COM_TARGET = 14.0       # measured on the MNIST test set: mean center of mass = (13.99, 14.01)
PEN = 0.06              # pen radius as a fraction of the drawing's longer side
WORK = 160              # resolution (px) of the drawing's longer side while rendering


def area_matrix(n_in, n_out):
    """(n_out, n_in) matrix that resizes a 1-D signal by area averaging."""
    m = np.zeros((n_out, n_in))
    step = n_in / n_out
    for i in range(n_out):
        a, b = i * step, (i + 1) * step
        for j in range(int(np.floor(a)), min(int(np.ceil(b)), n_in)):
            m[i, j] = (min(b, j + 1) - max(a, j)) / step
    return m


def area_resize(img, h_out, w_out):
    return area_matrix(img.shape[0], h_out) @ img @ area_matrix(img.shape[1], w_out).T


def _shift(img, dy, dx):
    """Shift a 2-D array by whole pixels, filling with zeros (no wrap-around)."""
    out = np.zeros_like(img)
    h, w = img.shape
    ys, yd = (slice(0, h - dy), slice(dy, h)) if dy >= 0 else (slice(-dy, h), slice(0, h + dy))
    xs, xd = (slice(0, w - dx), slice(dx, w)) if dx >= 0 else (slice(-dx, w), slice(0, w + dx))
    out[yd, xd] = img[ys, xs]
    return out


def to_mnist(raster, threshold=0.02):
    """Grayscale drawing (float, any size, 0 = background) -> uint8 28x28.

    Returns None if the drawing is empty.
    """
    raster = np.asarray(raster, dtype=np.float64)
    rows = np.nonzero((raster > threshold).any(axis=1))[0]
    cols = np.nonzero((raster > threshold).any(axis=0))[0]
    if len(rows) == 0:
        return None
    crop = raster[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]

    # Pad to a square (centered) so one resize keeps the aspect ratio.
    h, w = crop.shape
    s = max(h, w)
    square = np.zeros((s, s))
    square[(s - h) // 2:(s - h) // 2 + h, (s - w) // 2:(s - w) // 2 + w] = crop
    small = area_resize(square, DIGIT_BOX, DIGIT_BOX)

    img = np.zeros((FIELD, FIELD))
    off = (FIELD - DIGIT_BOX) // 2
    img[off:off + DIGIT_BOX, off:off + DIGIT_BOX] = small

    # Move the center of mass to (14, 14), in whole pixels like MNIST.
    total = img.sum()
    idx = np.arange(FIELD)
    cy = (img.sum(axis=1) * idx).sum() / total
    cx = (img.sum(axis=0) * idx).sum() / total
    img = _shift(img, int(round(COM_TARGET - cy)), int(round(COM_TARGET - cx)))

    img /= img.max()
    return np.clip(np.rint(img * 255), 0, 255).astype(np.uint8)


def _draw_segment(img, a, b, r):
    """Anti-aliased thick line from a to b (x, y in pixels), max-blended into img."""
    h, w = img.shape
    x0 = max(int(np.floor(min(a[0], b[0]) - r - 1)), 0)
    x1 = min(int(np.ceil(max(a[0], b[0]) + r + 1)), w)
    y0 = max(int(np.floor(min(a[1], b[1]) - r - 1)), 0)
    y1 = min(int(np.ceil(max(a[1], b[1]) + r + 1)), h)
    if x0 >= x1 or y0 >= y1:
        return
    yy, xx = np.mgrid[y0:y1, x0:x1] + 0.5
    ab = np.subtract(b, a)
    denom = float(ab @ ab)
    t = 0.0 if denom == 0 else np.clip(((xx - a[0]) * ab[0] + (yy - a[1]) * ab[1]) / denom, 0, 1)
    d = np.hypot(xx - (a[0] + t * ab[0]), yy - (a[1] + t * ab[1]))
    np.maximum(img[y0:y1, x0:x1], np.clip(r + 0.5 - d, 0, 1), out=img[y0:y1, x0:x1])


def render_strokes(strokes):
    """List of strokes, each a list of (x, y) points -> float raster (0..1).

    The drawing is rescaled so its longer side is WORK px and drawn with a pen
    of radius PEN * WORK, so the result does not depend on how big it was drawn.
    Returns None if there are no points.
    """
    strokes = [np.asarray(s, dtype=np.float64).reshape(-1, 2) for s in strokes if len(s)]
    if not strokes:
        return None
    pts = np.concatenate(strokes)
    lo = pts.min(axis=0)
    extent = max(float((pts.max(axis=0) - lo).max()), 1.0)
    scale = WORK / extent
    r = PEN * WORK
    pad = int(np.ceil(r)) + 2
    size = np.ceil((pts.max(axis=0) - lo) * scale).astype(int) + 2 * pad
    img = np.zeros((size[1], size[0]))
    for s in strokes:
        p = (s - lo) * scale + pad
        if len(p) == 1:
            _draw_segment(img, p[0], p[0], r)
        for a, b in zip(p[:-1], p[1:]):
            _draw_segment(img, a, b, r)
    return img


def strokes_to_mnist(strokes):
    raster = render_strokes(strokes)
    return None if raster is None else to_mnist(raster)
