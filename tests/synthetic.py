"""Synthetic 'hand-drawn' digits made of pen strokes, used by the tests."""
import numpy as np
from PIL import Image, ImageDraw


def _arc(cx, cy, rx, ry, a0, a1, n=40):
    t = np.radians(np.linspace(a0, a1, n))
    return list(zip(cx + rx * np.cos(t), cy - ry * np.sin(t)))


STROKES = {
    0: [_arc(0.5, 0.5, 0.25, 0.4, 0, 360)],
    1: [[(0.38, 0.25), (0.55, 0.1), (0.55, 0.9)]],
    2: [_arc(0.5, 0.32, 0.25, 0.22, 160, -30) + [(0.25, 0.9), (0.78, 0.9)]],
    3: [_arc(0.48, 0.3, 0.24, 0.2, 150, -90), _arc(0.48, 0.7, 0.27, 0.2, 90, -150)],
    4: [[(0.62, 0.1), (0.22, 0.62), (0.8, 0.62)], [(0.62, 0.35), (0.62, 0.9)]],
    5: [[(0.74, 0.1), (0.32, 0.1), (0.29, 0.45)] + _arc(0.48, 0.66, 0.26, 0.24, 120, -150)],
    6: [_arc(0.62, 0.45, 0.3, 0.35, 70, 180) + _arc(0.5, 0.7, 0.22, 0.2, 180, -180)],
    7: [[(0.24, 0.12), (0.78, 0.12), (0.42, 0.9)]],
    8: [_arc(0.5, 0.29, 0.2, 0.19, 0, 360), _arc(0.5, 0.7, 0.25, 0.21, 0, 360)],
    9: [_arc(0.5, 0.32, 0.22, 0.22, 0, 360) + [(0.72, 0.32), (0.66, 0.9)]],
}


def draw_digit(d, size=320, width=18, scale=0.8, angle=0.0, dx=0.0, dy=0.0):
    """Render digit d as white pen strokes on black, like the app's canvas."""
    img = Image.new("L", (size, size), 0)
    dr = ImageDraw.Draw(img)
    a = np.radians(angle)
    for stroke in STROKES[d]:
        pts = []
        for x, y in stroke:
            x, y = x - 0.5, y - 0.5
            x, y = x * np.cos(a) - y * np.sin(a), x * np.sin(a) + y * np.cos(a)
            pts.append(((0.5 + dx + x * scale) * size, (0.5 + dy + y * scale) * size))
        dr.line(pts, fill=255, width=width, joint="curve")
        r = width / 2
        for px, py in (pts[0], pts[-1]):
            dr.ellipse((px - r, py - r, px + r, py + r), fill=255)
    return img
