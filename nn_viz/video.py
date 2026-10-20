"""Export the forward pass of one input as an MP4 video."""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Callable

import matplotlib
import numpy as np
from matplotlib.animation import FFMpegWriter
from matplotlib.figure import Figure

from .model import Network
from .network_view import BAD, BG, GOOD, MUTED, TXT, NetworkView


def ensure_ffmpeg() -> None:
    """Use the system ffmpeg if installed, otherwise the one bundled with imageio-ffmpeg."""
    if shutil.which(matplotlib.rcParams["animation.ffmpeg_path"]) or shutil.which("ffmpeg"):
        return
    try:
        import imageio_ffmpeg
    except ImportError as e:
        raise RuntimeError("FFmpeg not found. Run `pip install imageio-ffmpeg` "
                           "or `conda install -c conda-forge ffmpeg`.") from e
    matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()


def export_forward_pass(net: Network, x: np.ndarray, out_path: str | Path,
                        true_label: int | None = None, fps: int = 30,
                        progress_cb: Callable[[float], None] | None = None) -> Path:
    """Render a 1920x1080 video of x travelling through the network."""
    ensure_ffmpeg()
    out_path = Path(out_path)
    fig = Figure(figsize=(19.2, 10.8), dpi=100, facecolor=BG)

    # left: the 8x8 input
    ax_img = fig.add_axes([0.03, 0.33, 0.17, 0.30])
    ax_img.imshow(x.reshape(8, 8), cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    ax_img.set_xticks([]); ax_img.set_yticks([])
    for s in ax_img.spines.values():
        s.set_color("#2a2c33")
    fig.text(0.115, 0.66, "Your digit as 8×8 pixels", ha="center", color=TXT, fontsize=15,
             fontweight="bold")

    ax = fig.add_axes([0.22, 0.1, 0.77, 0.82])
    view = NetworkView(ax, net)
    view.set_input(x)
    pred, probs = net.predict(x)

    title = fig.text(0.02, 0.95, "How the neural network reads your digit", color=TXT,
                     fontsize=20, fontweight="bold")
    fig.text(0.98, 0.955, f"{' → '.join(map(str, net.sizes))}   ·   {net.n_params:,} parameters",
             ha="right", color=MUTED, fontsize=13)
    caption = fig.text(0.02, 0.03, "", color=MUTED, fontsize=15)
    result = fig.text(0.6, 0.06, "", ha="center", fontsize=22, fontweight="bold")

    L = view.L
    hold_in, per_layer, hold_out = 1.5, 1.8, 3.0
    total = int((hold_in + per_layer * L + hold_out) * fps)
    steps = []
    for l in range(L):
        dest = "the output layer" if l == L - 1 else f"hidden layer {l + 1}"
        act = "softmax" if l == L - 1 else "ReLU"
        steps.append(f"Step {l + 2}  ·  signals flow along {net.weights[l].size:,} weighted links "
                     f"into {dest}, each neuron sums them and applies {act}")

    writer = FFMpegWriter(fps=fps, codec="libx264", bitrate=8000, extra_args=["-pix_fmt", "yuv420p"])
    with writer.saving(fig, str(out_path), dpi=100):
        for f in range(total):
            t = f / fps
            p = min(max(0.0, (t - hold_in) / per_layer), L + 1)
            view.show(p)
            if p <= 0:
                caption.set_text("Step 1  ·  the 64 pixel values enter the input neurons")
            elif p < L:
                caption.set_text(steps[int(p)])
            else:
                caption.set_text("Final step  ·  the output neuron with the highest probability "
                                 "is the prediction")
                txt = f"Prediction: {pred}   ({probs[pred] * 100:.1f}% confident)"
                color = TXT
                if true_label is not None:
                    ok = pred == true_label
                    txt += "   ✓ correct" if ok else f"   ✗ (you drew {true_label})"
                    color = GOOD if ok else BAD
                result.set_text(txt)
                result.set_color(color)
            writer.grab_frame()
            if progress_cb and f % 10 == 0:
                progress_cb(f / total)
    if progress_cb:
        progress_cb(1.0)
    return out_path
