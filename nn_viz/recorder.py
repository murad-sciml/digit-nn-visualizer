"""
Record the app window to an MP4 while the app is running.

The Tk main thread only publishes the window's position/size (cheap); a
background thread grabs that screen region and pipes frames into ffmpeg, so the
GUI and its animations don't stutter. Frames are paced by the wall clock, so the
video plays back in real time.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

from PIL import Image, ImageGrab


def ffmpeg_exe() -> str:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError as e:
        raise RuntimeError("FFmpeg not found. Run `pip install imageio-ffmpeg`.") from e


def _grab(bbox):
    if sys.platform == "win32":
        return ImageGrab.grab(bbox=bbox, all_screens=True)
    if sys.platform == "darwin":
        return ImageGrab.grab(bbox=bbox)
    return ImageGrab.grab(bbox=bbox, xdisplay=os.environ.get("DISPLAY"))


class ScreenRecorder:
    def __init__(self, root, out_dir: str | Path = "recordings", fps: int = 15, max_width: int = 1920):
        self.root = root
        self.fps = fps
        self.max_width = max_width
        self.out_dir = Path(out_dir)
        self.path: Path | None = None
        self.bbox = None
        self.visible = True
        self.started_at = 0.0
        self.frames = 0
        self.error: str | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._proc: subprocess.Popen | None = None
        self._size = None

    # ---- called on the Tk main thread ------------------------------------
    def start(self):
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.out_dir / f"session_{datetime.now():%Y%m%d_%H%M%S}.mp4"
        self.root.update_idletasks()
        self._publish_bbox()
        self.started_at = time.perf_counter()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _publish_bbox(self):
        """Keep the window position up to date (Tk may only be touched from this thread)."""
        if self._stop.is_set():
            return
        try:
            self.visible = self.root.state() != "iconic"
            x, y = self.root.winfo_rootx(), self.root.winfo_rooty()
            w, h = self.root.winfo_width(), self.root.winfo_height()
            if w > 10 and h > 10:
                self.bbox = (x, y, x + w, y + h)
        except Exception:
            return
        self.root.after(250, self._publish_bbox)

    def elapsed(self) -> float:
        return time.perf_counter() - self.started_at if self.started_at else 0.0

    def stop(self, timeout: float = 30) -> Path | None:
        """Stop recording and finish writing the file. Returns the saved path (or None)."""
        self._stop.set()
        if self._thread:
            self._thread.join(timeout)
        if self._proc:
            try:
                self._proc.stdin.close()
            except Exception:
                pass
            try:
                self._proc.wait(timeout)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        if self.frames == 0 or not (self.path and self.path.exists()):
            return None
        return self.path

    # ---- background thread ------------------------------------------------
    def _open_ffmpeg(self, w, h):
        cmd = [ffmpeg_exe(), "-y", "-loglevel", "error",
               "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(self.fps), "-i", "-",
               "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
               "-movflags", "+faststart", str(self.path)]
        flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                      creationflags=flags)

    def _frame(self):
        img = _grab(self.bbox).convert("RGB")
        if self._size is None:
            w, h = img.size
            if w > self.max_width:
                h, w = round(h * self.max_width / w), self.max_width
            self._size = (w - w % 2, h - h % 2)          # H.264 needs even dimensions
            self._open_ffmpeg(*self._size)
        if img.size != self._size:                      # window was resized
            img = img.resize(self._size, Image.Resampling.BILINEAR)
        return img.tobytes()

    def _run(self):
        last = None
        interval = 1.0 / self.fps
        try:
            while not self._stop.is_set():
                t = time.perf_counter()
                if self.bbox is not None and self.visible:
                    last = self._frame()
                if last is not None:
                    # keep real-time pace: write as many frames as wall-clock time requires
                    due = int((t - self.started_at) * self.fps) + 1
                    while self.frames < due:
                        self._proc.stdin.write(last)
                        self.frames += 1
                time.sleep(max(0.0, interval - (time.perf_counter() - t)))
        except (BrokenPipeError, OSError, ValueError) as e:
            self.error = str(e)
        except Exception as e:  # never crash the app because of the recorder
            self.error = repr(e)
