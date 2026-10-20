"""
Standalone demo video (python scripts/render_demo_video.py, ~6 min to render).

Deep neural network forward-pass animation on handwritten digits.
Architecture: 8x8 image -> 64 input -> 32 -> 24 -> 16 -> 10 outputs (ReLU hidden, softmax output).

The video shows: pixels flying from the image into the input neurons, signals travelling
along weighted links (pulse size = activation x weight), neurons computing and lighting up
layer by layer, and the final digit prediction with probability bars.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import shutil
if shutil.which("ffmpeg") is None:          # fall back to the ffmpeg bundled with imageio-ffmpeg
    import imageio_ffmpeg
    matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from matplotlib.collections import LineCollection
from matplotlib.patches import FancyBboxPatch, Rectangle
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

# ---------------- model ----------------
digits = load_digits()
X = digits.data / 16.0
y = digits.target
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)
HIDDEN = (32, 24, 16)
clf = MLPClassifier(hidden_layer_sizes=HIDDEN, activation="relu", alpha=1e-3,
                    max_iter=3000, random_state=1).fit(Xtr, ytr)
TEST_ACC = clf.score(Xte, yte)
print("test accuracy:", TEST_ACC)

W, B = clf.coefs_, clf.intercepts_
SIZES = [64, *HIDDEN, 10]
L = len(SIZES) - 1
N_PARAMS = sum(w.size + b.size for w, b in zip(W, B))

def forward(x):
    acts, zs = [x], []
    for l in range(L):
        z = acts[-1] @ W[l] + B[l]
        zs.append(z)
        if l < L - 1:
            acts.append(np.maximum(z, 0))
        else:
            e = np.exp(z - z.max()); acts.append(e / e.sum())
    return acts, zs

# ---------------- style / layout ----------------
BG, TXT, MUTED = "#0f1117", "#e8e6df", "#8b8a85"
POS, NEG = np.array([55, 138, 221]) / 255, np.array([216, 90, 48]) / 255
C_IN, C_HID, C_OUT = "#1D9E75", "#7F77DD", "#D4537E"
PULSE = "#FAC775"
FPS = 30
FW, FH = 19.2, 10.8

fig = plt.figure(figsize=(FW, FH), dpi=100)
fig.patch.set_facecolor(BG)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, FW); ax.set_ylim(0, FH); ax.axis("off")

XS = [4.3, 7.3, 10.1, 12.7, 15.3]
TOP, BOT = 8.95, 1.55
RADII = [0.045, 0.085, 0.11, 0.14, 0.23]

def ys(l):
    n = SIZES[l]
    span = (TOP - BOT) if l == 0 else min(TOP - BOT, (n - 1) * [0, 0.235, 0.3, 0.42, 0.72][l])
    mid = (TOP + BOT) / 2
    return mid + np.linspace(span / 2, -span / 2, n)

POS_XY = [np.column_stack([np.full(SIZES[l], XS[l]), ys(l)]) for l in range(L + 1)]

def hexrgb(h):
    h = h.lstrip("#"); return np.array([int(h[k:k + 2], 16) / 255 for k in (0, 2, 4)])
BGc = hexrgb(BG)

# --- input image (8x8 grid) ---
IMG_X0, IMG_Y0, CELL = 0.55, 3.55, 0.36
img_cells = []
for r in range(8):
    for c in range(8):
        rect = Rectangle((IMG_X0 + c * CELL, IMG_Y0 + (7 - r) * CELL), CELL * 0.92, CELL * 0.92,
                         fc=BG, ec="#2a2c33", lw=0.6, zorder=3)
        ax.add_patch(rect); img_cells.append(rect)
CELL_CENTERS = np.array([[IMG_X0 + c * CELL + CELL * 0.46, IMG_Y0 + (7 - r) * CELL + CELL * 0.46]
                         for r in range(8) for c in range(8)])
ax.text(IMG_X0 + 4 * CELL, IMG_Y0 + 8 * CELL + 0.35, "Input image (8×8 pixels)", ha="center",
        fontsize=13, color=TXT, fontweight="bold")
img_caption = ax.text(IMG_X0 + 4 * CELL, IMG_Y0 - 0.35, "", ha="center", fontsize=12, color=MUTED)

# --- edges as LineCollections (one per layer) ---
edge_cols, edge_w, edge_segs = [], [], []
for l in range(L):
    a, b = POS_XY[l], POS_XY[l + 1]
    ii, jj = np.meshgrid(np.arange(SIZES[l]), np.arange(SIZES[l + 1]), indexing="ij")
    ii, jj = ii.ravel(), jj.ravel()
    segs = np.stack([a[ii] + [RADII[l], 0], b[jj] - [RADII[l + 1], 0]], axis=1)
    w = W[l][ii, jj]
    lw = 0.25 + 1.6 * np.abs(w) / np.abs(w).max()
    lc = LineCollection(segs, linewidths=lw, zorder=1)
    ax.add_collection(lc)
    edge_cols.append(lc); edge_w.append(w); edge_segs.append(segs)

BASE_ALPHA = [0.035, 0.07, 0.1, 0.16]

def edge_colors(l, highlight=None):
    w = edge_w[l]
    rgb = np.where((w > 0)[:, None], POS, NEG)
    alpha = np.full(len(w), BASE_ALPHA[l])
    if highlight is not None:
        alpha = alpha + highlight * (0.85 - BASE_ALPHA[l])
    return np.column_stack([rgb, np.clip(alpha, 0, 1)])

# --- nodes (scatter per layer, sizes in points^2) ---
pts_per_unit = fig.dpi * 72 / 100 * (fig.get_size_inches()[0] / FW)
def r2s(r): return (2 * r * 72) ** 2  # radius in data units (1 unit = 1 inch here)

node_sc = []
for l in range(L + 1):
    edge = C_IN if l == 0 else C_OUT if l == L else C_HID
    sc = ax.scatter(POS_XY[l][:, 0], POS_XY[l][:, 1], s=r2s(RADII[l]), facecolors=BG,
                    edgecolors=edge, linewidths=1.2 if l else 0.7, zorder=4)
    node_sc.append(sc)

# --- headers ---
NAMES = ["Input layer", "Hidden 1", "Hidden 2", "Hidden 3", "Output layer"]
SUBS = ["64 pixels", "32 neurons · ReLU", "24 neurons · ReLU", "16 neurons · ReLU", "10 digits · softmax"]
for l in range(L + 1):
    ax.text(XS[l], 9.65, NAMES[l], ha="center", fontsize=14, color=TXT, fontweight="bold")
    ax.text(XS[l], 9.32, SUBS[l], ha="center", fontsize=10.5, color=MUTED)
for l in range(L):
    ax.text((XS[l] + XS[l + 1]) / 2, 1.2, f"{W[l].size + B[l].size:,} params",
            ha="center", fontsize=9.5, color=MUTED)

# --- output labels, values, bars ---
out_lbl, out_val, out_bars = [], [], []
for j in range(10):
    x, yy = POS_XY[L][j]
    out_val.append(ax.text(x, yy, "", ha="center", va="center", fontsize=9, color=TXT,
                           fontweight="bold", zorder=6))
    out_lbl.append(ax.text(x + 0.4, yy, str(j), ha="left", va="center", fontsize=15,
                           color=MUTED, fontweight="bold"))
    ax.add_patch(FancyBboxPatch((x + 0.75, yy - 0.08), 2.4, 0.16,
                 boxstyle="round,pad=0,rounding_size=0.06", fc="#22242b", ec="none", zorder=3))
    bar = FancyBboxPatch((x + 0.75, yy - 0.08), 0, 0.16,
                         boxstyle="round,pad=0,rounding_size=0.06", fc=C_OUT, ec="none", zorder=4)
    bar.set_visible(False); ax.add_patch(bar); out_bars.append(bar)

title = ax.text(0.5, 10.3, "", fontsize=17, color=TXT, fontweight="bold")
subtitle = ax.text(FW - 0.5, 10.3, f"{N_PARAMS:,} trainable parameters   ·   test accuracy {TEST_ACC*100:.1f}%",
                   fontsize=12, color=MUTED, ha="right")
caption = ax.text(0.5, 0.35, "", fontsize=13, color=MUTED)
formula = ax.text(FW / 2 + 0.6, 0.78, "", fontsize=13, color=PULSE, ha="center", family="monospace")
pred_box = ax.text(FW / 2 + 0.6, 0.8, "", fontsize=19, ha="center", va="center", fontweight="bold")
ax.text(FW - 0.5, 0.35, "blue = positive weight   orange = negative weight   pulse size = activation × weight",
        fontsize=10, color=MUTED, ha="right")

pulse_sc = ax.scatter([], [], s=[], c=PULSE, zorder=5, edgecolors="none")
fly_sc = ax.scatter([], [], s=[], marker="s", zorder=7, edgecolors="none")

# ---------------- helpers ----------------
def ease(t):
    t = np.clip(t, 0, 1); return t * t * (3 - 2 * t)

def fill_colors(l, vals, t):
    base = hexrgb(C_IN if l == 0 else C_OUT if l == L else C_HID)
    t = np.broadcast_to(np.asarray(t, float), vals.shape)
    k = (t * (0.12 + 0.88 * vals))[:, None]
    return BGc * (1 - k) + base * k

def set_nodes(l, vals, t):
    node_sc[l].set_facecolors(fill_colors(l, vals, t))
    pulse = 1 + 0.35 * np.sin(np.pi * np.clip(np.broadcast_to(t, vals.shape), 0, 1)) * vals
    node_sc[l].set_sizes(r2s(RADII[l]) * pulse ** 2)

def reset():
    for l in range(L + 1):
        node_sc[l].set_facecolors(BG); node_sc[l].set_sizes(np.full(SIZES[l], r2s(RADII[l])))
    for l in range(L):
        edge_cols[l].set_colors(edge_colors(l))
    for c in img_cells: c.set_facecolor(BG)
    for j in range(10):
        out_val[j].set_text(""); out_lbl[j].set_color(MUTED); out_bars[j].set_visible(False)
    for s in (pulse_sc, fly_sc):
        s.set_offsets(np.empty((0, 2))); s.set_sizes([])
    formula.set_text(""); pred_box.set_text(""); img_caption.set_text("")

def norm(v):
    m = v.max(); return v / m if m > 0 else v

# ---------------- per-sample animation ----------------
def sample_frames(idx, writer, k):
    x, label = Xte[idx], yte[idx]
    acts, zs = forward(x)
    pred = int(np.argmax(acts[-1]))
    reset()
    title.set_text(f"Sample {k}   ·   true digit: {label}")

    # 0. draw image
    caption.set_text("Step 1  ·  a handwritten digit arrives as an 8×8 grid of pixel brightness values")
    for f in range(int(1.0 * FPS)):
        t = ease(f / (0.7 * FPS))
        for p, c in enumerate(img_cells):
            c.set_facecolor(tuple(BGc * (1 - x[p] * t) + np.ones(3) * x[p] * t))
        writer.grab_frame()
    img_caption.set_text(f"true label: {label}")

    # 1. pixels fly into input neurons (staggered row by row)
    caption.set_text("Step 1  ·  each pixel is flattened into one input neuron (64 inputs)")
    n = int(1.5 * FPS)
    for f in range(n):
        tt = ease(f / (0.9 * FPS) - (np.arange(64) // 8) * 0.07)
        p = CELL_CENTERS + (POS_XY[0] - CELL_CENTERS) * tt[:, None]
        fly_sc.set_offsets(p)
        fly_sc.set_sizes(np.where(tt < 1, 60 * (1 - 0.6 * tt), 0))
        fly_sc.set_facecolors([(1, 1, 1, 0.25 + 0.75 * v) for v in x])
        arrived = (tt >= 1).astype(float)
        set_nodes(0, x, arrived)
        writer.grab_frame()
    fly_sc.set_offsets(np.empty((0, 2))); fly_sc.set_sizes([])

    # 2. layer by layer
    for l in range(L):
        dest = "the output layer" if l == L - 1 else f"hidden layer {l+1}"
        caption.set_text(f"Step {l+2}  ·  signals flow along {W[l].size:,} weighted links into {dest}")
        contrib = (acts[l][:, None] * W[l]).ravel()
        mag = np.abs(contrib) / (np.abs(contrib).max() + 1e-9)
        top = np.argsort(-mag)[: min(220, len(mag))]
        top = top[mag[top] > 0.02]
        segs = edge_segs[l]
        rng = np.random.default_rng(l + 10 * k)
        offs = rng.uniform(0, 0.45, len(top))
        dur = 0.55
        travel = int(1.6 * FPS)
        for f in range(travel):
            g = f / (travel - 1)
            edge_cols[l].set_colors(edge_colors(l, np.sin(np.pi * g) * mag))
            P, S = [], []
            for lag, fade in ((0.0, 1.0), (0.035, 0.55), (0.07, 0.3)):   # pulse + 2-dot trail
                tt = np.clip((g - lag - offs) / dur, 0, 1)
                live = (tt > 0) & (tt < 1)
                p = segs[top, 0] + (segs[top, 1] - segs[top, 0]) * tt[:, None]
                P.append(p[live]); S.append(((6 + 90 * mag[top]) * fade)[live])
            pulse_sc.set_offsets(np.vstack(P)); pulse_sc.set_sizes(np.concatenate(S))
            writer.grab_frame()
        pulse_sc.set_offsets(np.empty((0, 2))); pulse_sc.set_sizes([])
        edge_cols[l].set_colors(edge_colors(l))

        act_name = "softmax" if l == L - 1 else "ReLU"
        j0 = int(np.argmax(acts[l + 1]))
        formula.set_text(f"neuron {j0}:  z = Σ(activation × weight) + bias = {zs[l][j0]:+.2f}"
                         f"   →   {act_name}(z) = {acts[l+1][j0]:.2f}")
        caption.set_text(f"Step {l+2}  ·  each neuron sums its inputs plus bias, then applies {act_name}"
                         + ("  (negative sums are switched off → 0)" if l < L - 1 else ""))
        vals = norm(acts[l + 1]) if l < L - 1 else acts[l + 1]
        comp = int(0.9 * FPS)
        order = np.arange(SIZES[l + 1])
        for f in range(comp):
            t = ease(f / (0.45 * FPS) - order * (0.4 / SIZES[l + 1]))
            set_nodes(l + 1, vals, t)
            if l == L - 1:
                for j in range(10):
                    wv = 2.4 * acts[-1][j] * t[j]
                    out_bars[j].set_width(wv); out_bars[j].set_visible(wv > 0.03)
                    if t[j] > 0.5: out_val[j].set_text(f"{acts[-1][j]*100:.0f}")
            writer.grab_frame()
        formula.set_text("")

    # 3. prediction
    ok = pred == label
    caption.set_text("Final step  ·  the output neuron with the highest probability is the predicted digit")
    pred_box.set_text(f"Prediction: {pred}   ({acts[-1][pred]*100:.1f}% confident)   "
                      + ("✓ correct" if ok else f"✗ wrong (true {label})"))
    pred_box.set_color("#5DCAA5" if ok else "#F09595")
    out_lbl[pred].set_color(TXT)
    base = node_sc[L].get_sizes().copy()
    for f in range(int(2.4 * FPS)):
        s = base.copy(); s[pred] = r2s(RADII[L]) * (1.25 + 0.15 * np.sin(f / FPS * 6)) ** 2
        node_sc[L].set_sizes(s)
        writer.grab_frame()

def intro(writer):
    reset()
    title.set_text("Inside a deep neural network: recognising a handwritten digit")
    caption.set_text("64 inputs → 32 → 24 → 16 hidden neurons → 10 outputs   ·   "
                     "watch the signal travel from pixels to prediction")
    for _ in range(int(2.5 * FPS)):
        writer.grab_frame()

if __name__ == "__main__":
    probs = clf.predict_proba(Xte)
    picks = []
    for d in [3, 7, 0, 8]:                       # a few clear, correct examples
        cand = np.where((yte == d) & (clf.predict(Xte) == d))[0]
        picks.append(cand[np.argmax(probs[cand, d])] if d != 8 else cand[np.argsort(probs[cand, d])[len(cand)//2]])
    wrong = np.where(clf.predict(Xte) != yte)[0]
    if len(wrong): picks.append(wrong[0])        # show one mistake too
    writer = FFMpegWriter(fps=FPS, codec="libx264", bitrate=8000, extra_args=["-pix_fmt", "yuv420p"])
    with writer.saving(fig, "nn_deep_forward_pass.mp4", dpi=100):
        intro(writer)
        for k, idx in enumerate(picks, 1):
            sample_frames(idx, writer, k)
    print("saved nn_deep_forward_pass.mp4")
