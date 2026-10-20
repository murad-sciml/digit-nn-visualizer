"""
Model: a fully connected network 64 -> 32 -> 24 -> 16 -> 10 trained on the
scikit-learn handwritten digits dataset (8x8 images).

The trained weights are stored as a plain NumPy .npz file and the forward pass
is done in NumPy, so the saved model works with any scikit-learn version.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

HIDDEN = (32, 24, 16)
SIZES = (64, *HIDDEN, 10)
MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "digits_mlp.npz"


@dataclass
class Network:
    weights: list[np.ndarray]   # weights[l] has shape (n_in, n_out)
    biases: list[np.ndarray]
    test_accuracy: float = float("nan")

    @property
    def sizes(self) -> list[int]:
        return [self.weights[0].shape[0]] + [w.shape[1] for w in self.weights]

    @property
    def n_params(self) -> int:
        return int(sum(w.size + b.size for w, b in zip(self.weights, self.biases)))

    def forward(self, x: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Return (activations per layer incl. input, pre-activations z per layer)."""
        acts, zs = [np.asarray(x, dtype=float)], []
        last = len(self.weights) - 1
        for l, (w, b) in enumerate(zip(self.weights, self.biases)):
            z = acts[-1] @ w + b
            zs.append(z)
            if l < last:
                acts.append(np.maximum(z, 0.0))                     # ReLU
            else:
                e = np.exp(z - z.max())
                acts.append(e / e.sum())                            # softmax
        return acts, zs

    def predict(self, x: np.ndarray) -> tuple[int, np.ndarray]:
        probs = self.forward(x)[0][-1]
        return int(np.argmax(probs)), probs

    def save(self, path: Path = MODEL_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {f"W{l}": w for l, w in enumerate(self.weights)}
        arrays.update({f"b{l}": b for l, b in enumerate(self.biases)})
        np.savez_compressed(path, test_accuracy=self.test_accuracy, **arrays)

    @classmethod
    def load(cls, path: Path = MODEL_PATH) -> "Network":
        data = np.load(path)
        n = sum(1 for k in data.files if k.startswith("W"))
        return cls([data[f"W{l}"] for l in range(n)], [data[f"b{l}"] for l in range(n)],
                   float(data["test_accuracy"]))


def _augment(X: np.ndarray, y: np.ndarray, copies: int, seed: int = 0):
    """Small random rotations, shifts and zooms so the model copes with hand-drawn input."""
    from scipy import ndimage

    rng = np.random.default_rng(seed)
    out_x, out_y = [X], [y]
    imgs = X.reshape(-1, 8, 8)
    for _ in range(copies):
        batch = np.empty_like(imgs)
        for i, img in enumerate(imgs):
            big = ndimage.zoom(img, 4, order=1)                                    # 32x32
            big = ndimage.rotate(big, rng.uniform(-12, 12), reshape=False, order=1)
            z = rng.uniform(0.88, 1.08)
            zoomed = ndimage.zoom(big, z, order=1)
            canvas = np.zeros((32, 32))
            if zoomed.shape[0] >= 32:
                o = (zoomed.shape[0] - 32) // 2
                canvas = zoomed[o:o + 32, o:o + 32]
            else:
                o = (32 - zoomed.shape[0]) // 2
                canvas[o:o + zoomed.shape[0], o:o + zoomed.shape[1]] = zoomed
            canvas = ndimage.shift(canvas, rng.uniform(-2, 2, 2), order=1)
            batch[i] = canvas.reshape(8, 4, 8, 4).mean(axis=(1, 3))
        out_x.append(np.clip(batch.reshape(len(X), 64), 0, 1))
        out_y.append(y)
    return np.vstack(out_x), np.concatenate(out_y)


def train(save: bool = True, augment_copies: int = 4, verbose: bool = True) -> Network:
    """Train the network on sklearn's digits dataset and (optionally) save it."""
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split
    from sklearn.neural_network import MLPClassifier

    digits = load_digits()
    X, y = digits.data / 16.0, digits.target
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)
    if augment_copies:
        Xtr, ytr = _augment(Xtr, ytr, augment_copies)
    if verbose:
        print(f"Training {' -> '.join(map(str, SIZES))} network on {len(Xtr):,} images ...")
    clf = MLPClassifier(hidden_layer_sizes=HIDDEN, activation="relu", alpha=1e-3,
                        max_iter=400, random_state=1)
    clf.fit(Xtr, ytr)
    net = Network([w.copy() for w in clf.coefs_], [b.copy() for b in clf.intercepts_],
                  float(clf.score(Xte, yte)))
    if verbose:
        print(f"Test accuracy: {net.test_accuracy * 100:.2f}%  ({net.n_params:,} parameters)")
    if save:
        net.save()
        if verbose:
            print(f"Saved weights to {MODEL_PATH}")
    return net


def load_or_train() -> Network:
    if MODEL_PATH.exists():
        return Network.load()
    return train(save=True)
