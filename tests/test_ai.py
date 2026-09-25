import numpy as np
import pytest

from liggmath import ai, utils


def test_activations():
    x = np.array([-1.0, 0.0, 1.0])
    assert np.allclose(ai.sigmoid(np.array([0.0])), [0.5])
    assert np.allclose(ai.relu(x), [0.0, 0.0, 1.0])
    assert np.allclose(ai.relu_grad(x), [0.0, 0.0, 1.0])
    probs = ai.softmax(np.array([1.0, 1.0, 1.0]))
    assert np.allclose(probs, [1 / 3, 1 / 3, 1 / 3])
    assert probs.sum() == pytest.approx(1.0)


def test_activation_gradients_match_numerical():
    x = np.array([0.3, -0.7, 1.2])

    def f(v):
        return float(np.sum(ai.sigmoid(v)))

    analytic = ai.sigmoid_grad(x)
    numeric = utils.numerical_gradient(f, x.copy())
    assert np.allclose(analytic, numeric, atol=1e-5)


def test_losses():
    y_true = np.array([1.0, 0.0, 1.0])
    y_pred = np.array([1.0, 0.0, 1.0])
    assert ai.mse(y_true, y_pred) == pytest.approx(0.0)
    assert ai.mae(y_true, y_pred) == pytest.approx(0.0)
    assert ai.rmse(y_true, y_pred) == pytest.approx(0.0)

    y_pred_probs = np.array([0.9, 0.1, 0.8])
    bce = ai.binary_cross_entropy(y_true, y_pred_probs)
    assert bce > 0.0

    huber = ai.huber_loss(np.array([0.0]), np.array([10.0]), delta=1.0)
    assert huber == pytest.approx(10.0 - 0.5)


def test_metrics():
    y_true = np.array([1, 0, 1, 1])
    y_pred = np.array([1, 0, 0, 1])
    assert ai.accuracy(y_true, y_pred) == pytest.approx(0.75)
    assert ai.precision(y_true, y_pred) == pytest.approx(1.0)
    assert ai.recall(y_true, y_pred) == pytest.approx(2 / 3)
    cm = ai.confusion_matrix(y_true, y_pred, num_classes=2)
    assert cm.sum() == 4


def test_r2_score():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.0, 3.0])
    assert ai.r2_score(y_true, y_pred) == pytest.approx(1.0)


def test_init_shapes():
    for fn in (ai.xavier_uniform, ai.xavier_normal, ai.he_uniform, ai.he_normal):
        w = fn(4, 8, seed=0)
        assert w.shape == (4, 8)


def test_autograd_backward():
    a = ai.Value(2.0)
    b = ai.Value(-3.0)
    c = ai.Value(10.0)
    d = a * b + c
    e = d.relu()
    e.backward()
    assert e.data == pytest.approx(4.0)
    assert a.grad == pytest.approx(-3.0)
    assert b.grad == pytest.approx(2.0)
    assert c.grad == pytest.approx(1.0)


def test_optimizers_reduce_loss():
    w = np.array([5.0])
    opt = ai.SGD([w], lr=0.1)
    for _ in range(50):
        grad = 2 * w  # gradient of w^2
        opt.step([grad])
    assert abs(w[0]) < 0.1

    w2 = np.array([5.0])
    opt2 = ai.Adam([w2], lr=0.5)
    for _ in range(50):
        grad = 2 * w2
        opt2.step([grad])
    assert abs(w2[0]) < 0.5


def test_utils_one_hot_and_split():
    labels = np.array([0, 1, 2])
    oh = utils.one_hot(labels, num_classes=3)
    assert np.allclose(oh, np.eye(3))

    x = np.arange(10)
    y = np.arange(10)
    x_train, x_test, y_train, y_test = utils.train_test_split(x, y, test_size=0.3, seed=0)
    assert len(x_test) == 3
    assert len(x_train) == 7
    assert np.array_equal(x_train, y_train)


def test_clip_gradients():
    g = np.array([3.0, 4.0])
    clipped = utils.clip_gradients(g, max_norm=1.0)
    assert np.linalg.norm(clipped) == pytest.approx(1.0)
