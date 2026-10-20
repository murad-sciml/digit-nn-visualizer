"""
Digit NN Visualizer
Draw a digit with the mouse (or a finger / pen on a touchscreen) and watch a
neural network recognise it, layer by layer.

Run:  python app.py               (records the session to recordings/*.mp4)
      python app.py --no-record   (no recording)
"""
from __future__ import annotations

import argparse
import queue
import sys
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

from nn_viz.model import load_or_train  # noqa: E402
from nn_viz.network_view import BG, C_IN, MUTED, TXT, NetworkView  # noqa: E402
from nn_viz.preprocess import to_8x8  # noqa: E402
from nn_viz.recorder import ScreenRecorder  # noqa: E402

CANVAS = 320
RECORDINGS_DIR = Path(__file__).resolve().parent / "recordings"
REC_RED = "#E24B4A"
SECONDS_PER_LAYER = 1.0
PANEL = "#171a21"
BORDER = "#2a2c33"
ACCENT = "#7F77DD"
FONT = ("Segoe UI", 10) if sys.platform == "win32" else ("Helvetica", 10)


def font(size=10, bold=False):
    return (FONT[0], size, "bold") if bold else (FONT[0], size)


class App:
    def __init__(self, root: tk.Tk, record: bool = True, fps: int = 15):
        self.root = root
        root.title("Digit NN Visualizer")
        root.configure(bg=BG)
        root.minsize(1100, 700)

        self.net = load_or_train()
        self.x = None
        self.anim_job = None
        self.p = 0.0
        self.last = None
        self.export_q: queue.Queue = queue.Queue()

        self.brush = tk.IntVar(value=22)
        self.animate = tk.BooleanVar(value=True)
        self.speed = tk.DoubleVar(value=1.0)

        self._build_ui()
        self._bind_keys()
        self.clear()

        root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.rec = None
        if record:
            self.rec = ScreenRecorder(root, RECORDINGS_DIR, fps=fps)
            # start once the window is actually on screen
            root.after(600, self._start_recording)

    # ------------------------------------------------------------ recording
    def _start_recording(self):
        try:
            self.rec.start()
        except Exception as e:
            self.rec = None
            self.rec_lbl.configure(text="")
            self.status.configure(text=f"Screen recording unavailable: {e}")
            return
        self._update_rec_label()

    def _update_rec_label(self):
        if self.rec is None:
            return
        if self.rec.error:
            self.rec_lbl.configure(text="REC stopped", fg=MUTED)
            return
        t = int(self.rec.elapsed())
        dot = "●" if t % 2 == 0 else "○"
        self.rec_lbl.configure(text=f"{dot} REC {t // 60:02d}:{t % 60:02d}", fg=REC_RED)
        self.root.after(500, self._update_rec_label)

    def on_close(self):
        self._stop_anim()
        if self.rec is not None:
            self.status.configure(text="Saving recording …")
            self.rec_lbl.configure(text="● saving", fg=REC_RED)
            self.root.update_idletasks()
            path = self.rec.stop()
            if path:
                print(f"Session recording saved: {path}  ({self.rec.elapsed():.0f} s)")
            elif self.rec.error:
                print(f"Recording failed: {self.rec.error}")
            self.rec = None
        try:
            self.root.destroy()
        except tk.TclError:
            pass

    # ------------------------------------------------------------------ UI
    def _build_ui(self):
        top = tk.Frame(self.root, bg=BG)
        top.pack(fill="x", padx=16, pady=(12, 4))
        tk.Label(top, text="Draw a digit, watch the network think", bg=BG, fg=TXT,
                 font=font(16, True)).pack(side="left")
        self.rec_lbl = tk.Label(top, text="", bg=BG, fg=REC_RED, font=font(10, True))
        self.rec_lbl.pack(side="left", padx=(16, 0))
        acc = self.net.test_accuracy
        tk.Label(top, text=f"{' → '.join(map(str, self.net.sizes))}   ·   {self.net.n_params:,} parameters"
                           f"   ·   test accuracy {acc * 100:.1f}%",
                 bg=BG, fg=MUTED, font=font(10)).pack(side="right")

        body = tk.Frame(self.root, bg=BG)
        body.pack(fill="both", expand=True, padx=16, pady=8)

        # ---- left column
        left = tk.Frame(body, bg=BG)
        left.pack(side="left", fill="y")

        tk.Label(left, text="Your drawing", bg=BG, fg=TXT, font=font(11, True)).pack(anchor="w")
        self.cv = tk.Canvas(left, width=CANVAS, height=CANVAS, bg="black", highlightthickness=1,
                            highlightbackground=BORDER, cursor="pencil")
        self.cv.pack(pady=(4, 6))
        self.cv.bind("<ButtonPress-1>", self.on_down)
        self.cv.bind("<B1-Motion>", self.on_move)
        self.cv.bind("<ButtonRelease-1>", self.on_up)

        btns = tk.Frame(left, bg=BG)
        btns.pack(fill="x")
        self._button(btns, "Predict  ⏎", self.predict).pack(side="left", expand=True, fill="x")
        self._button(btns, "Clear  C", self.clear).pack(side="left", expand=True, fill="x", padx=6)
        self._button(btns, "Export MP4", self.export_video).pack(side="left", expand=True, fill="x")

        opts = tk.Frame(left, bg=BG)
        opts.pack(fill="x", pady=(8, 0))
        self._slider(opts, "Brush", self.brush, 8, 40, 1).pack(fill="x")
        self._slider(opts, "Speed", self.speed, 0.25, 3, 0.25).pack(fill="x")
        tk.Checkbutton(opts, text="Animate the forward pass", variable=self.animate, bg=BG, fg=TXT,
                       selectcolor=PANEL, activebackground=BG, activeforeground=TXT,
                       font=font(10)).pack(anchor="w", pady=(4, 0))

        seen = tk.Frame(left, bg=BG)
        seen.pack(fill="x", pady=(12, 0))
        col = tk.Frame(seen, bg=BG)
        col.pack(side="left")
        tk.Label(col, text="What the network sees (8×8)", bg=BG, fg=TXT,
                 font=font(10, True)).pack(anchor="w")
        self.grid = tk.Canvas(col, width=136, height=136, bg="black", highlightthickness=1,
                              highlightbackground=BORDER)
        self.grid.pack(anchor="w", pady=4)
        self.cells = [[self.grid.create_rectangle(c * 17, r * 17, c * 17 + 17, r * 17 + 17,
                                                  fill="black", outline="#1c1e24")
                       for c in range(8)] for r in range(8)]
        res = tk.Frame(seen, bg=BG)
        res.pack(side="left", fill="both", expand=True, padx=(14, 0))
        tk.Label(res, text="Prediction", bg=BG, fg=MUTED, font=font(10)).pack(anchor="w")
        self.pred_lbl = tk.Label(res, text="–", bg=BG, fg=TXT, font=font(54, True))
        self.pred_lbl.pack(anchor="w")
        self.conf_lbl = tk.Label(res, text="", bg=BG, fg=MUTED, font=font(10))
        self.conf_lbl.pack(anchor="w")

        # ---- right: the network
        right = tk.Frame(body, bg=BG, highlightthickness=1, highlightbackground=BORDER)
        right.pack(side="left", fill="both", expand=True, padx=(16, 0))
        self.fig = Figure(figsize=(9, 6.2), dpi=100, facecolor=BG)
        ax = self.fig.add_axes([0.01, 0.01, 0.98, 0.98])
        self.mpl = FigureCanvasTkAgg(self.fig, master=right)
        self.mpl.get_tk_widget().configure(bg=BG, highlightthickness=0)
        self.mpl.get_tk_widget().pack(fill="both", expand=True)
        self.mpl.draw()
        self.view = NetworkView(ax, self.net)
        self.mpl.draw()

        self.status = tk.Label(self.root, text="", bg=BG, fg=MUTED, font=font(9), anchor="w")
        self.status.pack(fill="x", padx=16, pady=(0, 8))

    def _button(self, parent, text, cmd):
        return tk.Button(parent, text=text, command=cmd, bg=PANEL, fg=TXT, activebackground=ACCENT,
                         activeforeground="white", relief="flat", bd=0, padx=8, pady=7,
                         font=font(10, True), highlightthickness=1, highlightbackground=BORDER,
                         cursor="hand2")

    def _slider(self, parent, label, var, lo, hi, step):
        f = tk.Frame(parent, bg=BG)
        tk.Label(f, text=label, width=6, anchor="w", bg=BG, fg=MUTED, font=font(10)).pack(side="left")
        tk.Label(f, textvariable=var, width=4, anchor="e", bg=BG, fg=TXT, font=font(10)).pack(
            side="right")
        tk.Scale(f, variable=var, from_=lo, to=hi, resolution=step, orient="horizontal",
                 showvalue=False, bg=ACCENT, troughcolor=BORDER, highlightthickness=0,
                 activebackground=TXT, bd=0, sliderrelief="flat", width=12, sliderlength=18).pack(
            side="left", fill="x", expand=True, padx=(0, 6))
        return f

    def _bind_keys(self):
        self.root.bind("<Return>", lambda e: self.predict())
        self.root.bind("<KeyPress-c>", lambda e: self.clear())
        self.root.bind("<Escape>", lambda e: self.clear())

    # -------------------------------------------------------------- drawing
    def on_down(self, e):
        self._stop_anim()
        self.cv.delete("hint")
        self.last = (e.x, e.y)
        self._dot(e.x, e.y)

    def on_move(self, e):
        if self.last is None:
            self.last = (e.x, e.y)
        x0, y0 = self.last
        r = self.brush.get()
        self.cv.create_line(x0, y0, e.x, e.y, fill="white", width=r, capstyle="round",
                            joinstyle="round", smooth=True)
        self.draw.line([x0, y0, e.x, e.y], fill=255, width=r)
        self._dot(e.x, e.y, canvas=False)
        self.last = (e.x, e.y)

    def on_up(self, _e):
        self.last = None
        self.predict()

    def _dot(self, x, y, canvas=True):
        r = self.brush.get() / 2
        if canvas:
            self.cv.create_oval(x - r, y - r, x + r, y + r, fill="white", outline="white")
        self.draw.ellipse([x - r, y - r, x + r, y + r], fill=255)

    def clear(self):
        self._stop_anim()
        self.cv.delete("all")
        self.img = Image.new("L", (CANVAS, CANVAS), 0)
        self.draw = ImageDraw.Draw(self.img)
        self.x = None
        for row in self.cells:
            for cell in row:
                self.grid.itemconfigure(cell, fill="black")
        self.pred_lbl.configure(text="–", fg=TXT)
        self.conf_lbl.configure(text="")
        self.cv.create_text(CANVAS / 2, CANVAS / 2, text="draw a digit 0–9 here",
                            fill="#4a4c55", font=font(12), tags="hint")
        self.view.clear()
        self.mpl.draw_idle()
        self.status.configure(text="Tip: draw one large digit that fills the box, like a printed "
                                   "number. Enter = predict, C = clear.")

    # ------------------------------------------------------------ predict
    def predict(self):
        self.cv.delete("hint")
        x = to_8x8(self.img)
        if x is None:
            self.status.configure(text="Draw something first.")
            return
        self.x = x
        for r in range(8):
            for c in range(8):
                v = int(x[r * 8 + c] * 255)
                self.grid.itemconfigure(self.cells[r][c], fill=f"#{v:02x}{v:02x}{v:02x}")
        pred, probs = self.net.predict(x)
        self.pred_lbl.configure(text=str(pred), fg=C_IN)
        second = int(np.argsort(probs)[-2])
        self.conf_lbl.configure(text=f"{probs[pred] * 100:.1f}% confident"
                                     f"   ·   next: {second} ({probs[second] * 100:.1f}%)")
        self.view.set_input(x)
        self._stop_anim()
        if self.animate.get():
            self.t0 = time.perf_counter()
            self.status.configure(text="Signals are flowing through the network …")
            self._tick()
        else:
            self.view.show(self.view.L + 1)
            self.mpl.draw_idle()
            self.status.configure(text=f"Predicted {pred}.")

    def _tick(self):
        # time-based so a full pass takes ~4.5 s at speed 1 regardless of redraw speed
        self.p = (time.perf_counter() - self.t0) * self.speed.get() / SECONDS_PER_LAYER
        self.view.show(self.p)
        self.mpl.draw_idle()
        if self.p >= self.view.L + 0.2:
            self.anim_job = None
            self.status.configure(text="Done. Draw another digit, or export this one as a video.")
            return
        self.anim_job = self.root.after(15, self._tick)

    def _stop_anim(self):
        if self.anim_job is not None:
            self.root.after_cancel(self.anim_job)
            self.anim_job = None

    # -------------------------------------------------------------- export
    def export_video(self):
        if self.x is None:
            messagebox.showinfo("Export MP4", "Draw a digit first.")
            return
        pred, _ = self.net.predict(self.x)
        path = filedialog.asksaveasfilename(defaultextension=".mp4", initialfile=f"digit_{pred}.mp4",
                                            filetypes=[("MP4 video", "*.mp4")])
        if not path:
            return
        x = self.x.copy()

        def work():
            try:
                from nn_viz.video import export_forward_pass
                export_forward_pass(self.net, x, path,
                                    progress_cb=lambda f: self.export_q.put(("progress", f)))
                self.export_q.put(("done", path))
            except Exception as exc:  # report any failure back to the UI thread
                self.export_q.put(("error", str(exc)))

        threading.Thread(target=work, daemon=True).start()
        self.status.configure(text="Rendering video … 0%")
        self.root.after(200, self._poll_export)

    def _poll_export(self):
        try:
            while True:
                kind, val = self.export_q.get_nowait()
                if kind == "progress":
                    self.status.configure(text=f"Rendering video … {val * 100:.0f}%")
                elif kind == "done":
                    self.status.configure(text=f"Saved video to {val}")
                    return
                else:
                    self.status.configure(text="Video export failed.")
                    messagebox.showerror("Export failed", val)
                    return
        except queue.Empty:
            pass
        self.root.after(200, self._poll_export)


def main():
    if sys.platform == "win32":
        try:  # crisp rendering on high-DPI Windows screens
            import ctypes
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except Exception:
            pass
    ap = argparse.ArgumentParser(description="Draw a digit and watch a neural network recognise it.")
    ap.add_argument("--no-record", action="store_true", help="don't record the session to video")
    ap.add_argument("--fps", type=int, default=15, help="recording frame rate (default 15)")
    args = ap.parse_args()

    root = tk.Tk()
    app = App(root, record=not args.no_record, fps=args.fps)
    try:
        root.mainloop()
    except KeyboardInterrupt:          # Ctrl+C in the terminal still saves the video
        app.on_close()


if __name__ == "__main__":
    main()
