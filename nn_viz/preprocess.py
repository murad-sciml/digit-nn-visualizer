"""
Convert a free-hand drawing into the 8x8 format of the scikit-learn digits dataset.

That dataset (UCI "optdigits") was made by scaling each digit to a 32x32 bitmap
and counting the "on" pixels in each 4x4 block, giving 8x8 values 0..16.
In that data every digit fills the full height of the box and strokes are thick
(roughly 30% of the 8x8 grid is ink for every digit), so we reproduce both:

1. crop the drawing to its bounding box
2. scale it so its height fills the 32x32 bitmap (keeping the aspect ratio)
3. binarise, then thicken thin strokes until ink coverage matches the dataset
4. average 4x4 blocks -> 8x8 values in [0, 1]
"""
from __future__ import annotations

import numpy as np
from PIL import Image
from scipy import ndimage

SIZE = 32               # bitmap size used by the original dataset
TARGET_INK = 0.27       # stroke thickening stops once this share of the 8x8 grid is ink
MAX_DILATE = 4


def to_8x8(img: Image.Image, return_bitmap: bool = False):
    """
    img: grayscale image, ink is bright (255) on a dark (0) background.
    Returns 64 values in [0, 1] (row-major 8x8), or None if nothing is drawn.
    With return_bitmap=True also returns the intermediate 32x32 bitmap.
    """
    arr = np.asarray(img.convert("L"), dtype=float) / 255.0
    ys, xs = np.nonzero(arr > 0.1)
    if len(ys) < 5:
        return (None, None) if return_bitmap else None

    crop = arr[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = crop.shape
    scale = min(SIZE / h, SIZE / w)
    nh, nw = max(1, round(h * scale)), max(1, round(w * scale))
    small = np.asarray(Image.fromarray((crop * 255).astype(np.uint8)).resize(
        (nw, nh), Image.Resampling.BOX), dtype=float) / 255.0

    bitmap = np.zeros((SIZE, SIZE), dtype=bool)
    top, left = (SIZE - nh) // 2, (SIZE - nw) // 2
    bitmap[top:top + nh, left:left + nw] = small > 0.2

    for _ in range(MAX_DILATE):
        if bitmap.reshape(8, 4, 8, 4).mean(axis=(1, 3)).mean() >= TARGET_INK:
            break
        bitmap = ndimage.binary_dilation(bitmap)

    x = bitmap.reshape(8, 4, 8, 4).mean(axis=(1, 3)).ravel()
    return (x, bitmap) if return_bitmap else x
