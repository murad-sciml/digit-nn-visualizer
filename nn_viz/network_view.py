"""
Matplotlib renderer for the network. The same class drives the live view in the
GUI and the frames of the exported MP4.

Animation is controlled by a single number `progress` p:
    p = 0         input layer is lit
    l < p < l+1   signals travel from layer l to layer l+1, which lights up near the end
    p >= L        output probabilities are shown (L = number of weight layers)
"""
from __future__ import annotations

import numpy as np
from matplotlib.collections import LineCollection

from .model import Network

BG = "#0f1117"
TXT = "#e8e6df"
MUTED = "#8b8a85"
POS = np.array([55, 138, 221]) / 255     # positive weight (blue)
NEG = np.array([216, 90, 48]) / 255      # negative weight (orange)
C_IN, C_HID, C_OUT = "#1D9E75", "#7F77DD", "#D4537E"
PULSE = "#FAC775"
GOOD, BAD = "#5DCAA5", "#F09595"

TRAVEL = 0.8          # share of each layer step spent on travelling pulses
MAX_PULSES = 220      # strongest signals drawn as pulses per layer


def hexrgb(h: str) -> np.ndarray:
    h = h.lstrip("#")
    return np.array([int(h[k:k + 2], 16) / 255 for k in (0, 2, 4)])


def ease(t):
    t = np.clip(t, 0, 1)
    return t * t * (3 - 2 * t)


class NetworkView:
    XLIM, YLIM = 16.0, 10.0

    def __init__(self, ax, net: Network, show_bars: bool = True, show_headers: bool = True):
        self.ax, self.net = ax, net
        self.sizes = net.sizes
        self.L = len(self.sizes) - 1
        self.show_bars = show_bars
        ax.set_facecolor(BG)
        ax.set_xlim(0, self.XLIM)
        ax.set_ylim(0, self.YLIM)
        ax.axis("off")

        right = 11.2 if show_bars else 14.8
        self.xs = np.linspace(1.0, right, self.L + 1)
        self.radii = [0.05, 0.1, 0.13, 0.17, 0.27][: self.L + 1]
        if len(self.radii) < self.L + 1:
            self.radii += [0.15] * (self.L + 1 - len(self.radii))
        top, bot = 8.7, 0.6
        self.ys = []
        for l, n in enumerate(self.sizes):
            gap = (top - bot) / max(n - 1, 1) if l == 0 else min((top - bot) / max(n - 1, 1),
                                                                    [0, 0.27, 0.34, 0.5, 0.86][min(l, 4)])
            span = gap * (n - 1)
            mid = (top + bot) / 2
            self.ys.append(mid + np.linspace(span / 2, -span / 2, n))

        # edges
        self.edges, self.edge_rgb, self.edge_lw = [], [], []
        self.base_alpha = []
        for l in range(self.L):
            w = net.weights[l]
            n_in, n_out = w.shape
            ii, jj = np.meshgrid(np.arange(n_in), np.arange(n_out), indexing="ij")
            self._ii_jj = getattr(self, "_ii_jj", []) + [(ii.ravel(), jj.ravel())]
            wv = w.ravel()
            lw = 0.2 + 1.5 * np.abs(wv) / np.abs(wv).max()
            lc = LineCollection(np.zeros((len(wv), 2, 2)), linewidths=lw, zorder=1)
            ax.add_collection(lc)
            self.edges.append(lc)
            self.edge_rgb.append(np.where((wv > 0)[:, None], POS, NEG))
            self.base_alpha.append(min(0.2, 3.0 / np.sqrt(len(wv))))

        # nodes
        self.nodes = []
        for l in range(self.L + 1):
            col = C_IN if l == 0 else C_OUT if l == self.L else C_HID
            sc = ax.scatter(np.full(self.sizes[l], self.xs[l]), self.ys[l], s=10,
                            facecolors=BG, edgecolors=col, linewidths=1.0 if l else 0.6, zorder=4)
            self.nodes.append(sc)

        self.pulses = ax.scatter([], [], s=[], c=PULSE, edgecolors="none", zorder=5)

        # headers
        self.texts = []
        if show_headers:
            names = ["Input"] + [f"Hidden {i}" for i in range(1, self.L)] + ["Output"]
            subs = [f"{self.sizes[0]} pixels"] + [f"{n} · ReLU" for n in self.sizes[1:-1]] + \
                   [f"{self.sizes[-1]} · softmax"]
            for l in range(self.L + 1):
                self.texts.append(ax.text(self.xs[l], 9.65, names[l], ha="center", va="center",
                                          fontsize=11, color=TXT, fontweight="bold"))
                self.texts.append(ax.text(self.xs[l], 9.25, subs[l], ha="center", va="center",
                                          fontsize=8.5, color=MUTED))

        # output labels and probability bars
        self.out_val, self.out_lbl, self.bar_bg, self.bars = [], [], [], []
        for j in range(self.sizes[-1]):
            y = self.ys[-1][j]
            self.out_val.append(ax.text(self.xs[-1], y, "", ha="center", va="center",
                                        fontsize=7.5, color=TXT, fontweight="bold", zorder=6))
            if show_bars:
                self.out_lbl.append(ax.text(self.xs[-1] + 0.55, y, str(j), ha="left", va="center",
                                            fontsize=12, color=MUTED, fontweight="bold"))
                self.bar_bg.append(ax.barh(y, 3.3, left=12.0, height=0.2, color="#22242b", zorder=2)[0])
                self.bars.append(ax.barh(y, 0, left=12.0, height=0.2, color=C_OUT, zorder=3)[0])

        self.acts = None
        self.contrib = [None] * self.L
        self._offsets = [np.random.default_rng(l).uniform(0, 0.4, MAX_PULSES) for l in range(self.L)]
        self.relayout()
        ax.figure.canvas.mpl_connect("resize_event", lambda e: self.relayout())
        self.clear()

    # ---------- geometry (depends on the axes' size in pixels) ----------
    def relayout(self):
        bbox = self.ax.get_window_extent()
        if bbox.width < 2 or bbox.height < 2:
            return
        self.ppu_y = bbox.height / self.YLIM * 72 / self.ax.figure.dpi      # points per data unit
        ppu_x = bbox.width / self.XLIM * 72 / self.ax.figure.dpi
        self.rx = [r * self.ppu_y / ppu_x for r in self.radii]                # radius in x units
        self.segs = []
        for l in range(self.L):
            ii, jj = self._ii_jj[l]
            a = np.column_stack([np.full(len(ii), self.xs[l] + self.rx[l]), self.ys[l][ii]])
            b = np.column_stack([np.full(len(jj), self.xs[l + 1] - self.rx[l + 1]), self.ys[l + 1][jj]])
            seg = np.stack([a, b], axis=1)
            self.edges[l].set_segments(seg)
            self.segs.append(seg)
        self.base_sizes = [(2 * r * self.ppu_y) ** 2 for r in self.radii]
        for l, sc in enumerate(self.nodes):
            sc.set_sizes(np.full(self.sizes[l], self.base_sizes[l]))

    # ---------- state ----------
    def clear(self):
        self.acts = None
        for l, sc in enumerate(self.nodes):
            sc.set_facecolors(BG)
            sc.set_sizes(np.full(self.sizes[l], self.base_sizes[l]))
        for l in range(self.L):
            self._edge_alpha(l, None)
        self.pulses.set_offsets(np.empty((0, 2)))
        self.pulses.set_sizes([])
        for t in self.out_val:
            t.set_text("")
        for t in self.out_lbl:
            t.set_color(MUTED)
        for b in self.bars:
            b.set_width(0)

    def set_input(self, x: np.ndarray):
        """Run the forward pass for x (64 values in [0, 1]) and prepare the animation."""
        self.acts, self.zs = self.net.forward(x)
        for l in range(self.L):
            c = (self.acts[l][:, None] * self.net.weights[l]).ravel()
            mag = np.abs(c) / (np.abs(c).max() + 1e-9)
            top = np.argsort(-mag)[:MAX_PULSES]
            self.contrib[l] = (mag, top[mag[top] > 0.03])

    def _edge_alpha(self, l, highlight):
        a = np.full(len(self.edge_rgb[l]), self.base_alpha[l])
        if highlight is not None:
            a = a + highlight * (0.9 - self.base_alpha[l])
        self.edges[l].set_colors(np.column_stack([self.edge_rgb[l], np.clip(a, 0, 1)]))

    def _light(self, l, t):
        vals = self.acts[l]
        if l == self.L:
            v = vals
        else:
            m = vals.max()
            v = vals / m if m > 0 else vals
        base = hexrgb(C_IN if l == 0 else C_OUT if l == self.L else C_HID)
        k = (np.clip(t, 0, 1) * (0.12 + 0.88 * v))[:, None]
        self.nodes[l].set_facecolors(hexrgb(BG) * (1 - k) + base * k)
        grow = 1 + 0.3 * np.sin(np.pi * np.clip(t, 0, 1)) * v
        self.nodes[l].set_sizes(self.base_sizes[l] * grow ** 2)

    def show(self, p: float):
        """Draw the animation state at progress p (see module docstring)."""
        if self.acts is None:
            return
        L = self.L
        p = float(np.clip(p, 0, L + 1))
        pts, sz = [], []
        self._light(0, 1.0)
        for l in range(1, L + 1):
            t = (p - (l - 1) - TRAVEL) / (1 - TRAVEL)      # 0..1 while layer l switches on
            if t <= 0:
                self.nodes[l].set_facecolors(BG)
                self.nodes[l].set_sizes(np.full(self.sizes[l], self.base_sizes[l]))
            else:
                self._light(l, min(t, 1.0))
        for l in range(L):
            mag, top = self.contrib[l]
            g = (p - l) / TRAVEL
            if 0 < g < 1:
                self._edge_alpha(l, np.sin(np.pi * g) * mag)
                offs = self._offsets[l][: len(top)]
                seg = self.segs[l][top]
                for lag, fade in ((0.0, 1.0), (0.04, 0.55), (0.08, 0.3)):
                    tt = np.clip((g - lag - offs) / 0.58, 0, 1)
                    live = (tt > 0) & (tt < 1)
                    pos = seg[:, 0] + (seg[:, 1] - seg[:, 0]) * tt[:, None]
                    pts.append(pos[live])
                    sz.append(((4 + 70 * mag[top]) * fade * (self.ppu_y / 50) ** 2)[live])
            else:
                self._edge_alpha(l, None)
        if pts:
            self.pulses.set_offsets(np.vstack(pts))
            self.pulses.set_sizes(np.concatenate(sz))
        else:
            self.pulses.set_offsets(np.empty((0, 2)))
            self.pulses.set_sizes([])

        probs = self.acts[-1]
        t_out = np.clip((p - (L - 1) - TRAVEL) / (1 - TRAVEL), 0, 1)
        pred = int(np.argmax(probs))
        for j, pr in enumerate(probs):
            self.out_val[j].set_text(f"{pr * 100:.0f}" if t_out > 0.5 else "")
            if self.show_bars:
                self.bars[j].set_width(3.3 * pr * ease(t_out))
                self.out_lbl[j].set_color(TXT if (t_out >= 1 and j == pred) else MUTED)
        if p >= L:
            s = np.full(self.sizes[-1], self.base_sizes[-1])
            s[pred] *= 1.35 ** 2
            self.nodes[-1].set_sizes(s)
