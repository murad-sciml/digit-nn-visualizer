import numpy as np
import pytest

from nn_viz.model import MODEL_PATH, SIZES, Network
from nn_viz.preprocess import to_8x8
from synthetic import draw_digit


@pytest.fixture(scope="module")
def net():
    if not MODEL_PATH.exists():
        pytest.skip("run `python train.py` first")
    return Network.load()


def test_architecture(net):
    assert net.sizes == list(SIZES)
    assert net.n_params == 3442


def test_forward_shapes_and_softmax(net):
    acts, zs = net.forward(np.random.default_rng(0).random(64))
    assert [len(a) for a in acts] == list(SIZES)
    assert np.isclose(acts[-1].sum(), 1.0)
    assert all((a >= 0).all() for a in acts[1:-1])          # ReLU outputs


def test_matches_sklearn_test_accuracy(net):
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split
    d = load_digits()
    _, Xte, _, yte = train_test_split(d.data / 16, d.target, test_size=0.25, random_state=0,
                                      stratify=d.target)
    acc = np.mean([net.predict(x)[0] == t for x, t in zip(Xte, yte)])
    assert acc > 0.95
    assert np.isclose(acc, net.test_accuracy)


def test_recognises_drawn_digits(net):
    correct = sum(net.predict(to_8x8(draw_digit(d)))[0] == d for d in range(10))
    assert correct >= 8
