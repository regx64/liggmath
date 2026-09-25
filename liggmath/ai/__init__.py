"""AI math utilities: activations, losses, metrics, autograd, optimizers, init."""

from .activations import (
    sigmoid,
    sigmoid_grad,
    relu,
    relu_grad,
    leaky_relu,
    leaky_relu_grad,
    tanh,
    tanh_grad,
    softmax,
    gelu,
    silu,
    elu,
)
from .losses import (
    mse,
    mae,
    rmse,
    binary_cross_entropy,
    categorical_cross_entropy,
    huber_loss,
)
from .metrics import (
    accuracy,
    precision,
    recall,
    f1_score,
    confusion_matrix,
    r2_score,
)
from .init import xavier_uniform, xavier_normal, he_uniform, he_normal
from .autograd import Value
from .optim import SGD, Momentum, Adam

__all__ = [
    "sigmoid",
    "sigmoid_grad",
    "relu",
    "relu_grad",
    "leaky_relu",
    "leaky_relu_grad",
    "tanh",
    "tanh_grad",
    "softmax",
    "gelu",
    "silu",
    "elu",
    "mse",
    "mae",
    "rmse",
    "binary_cross_entropy",
    "categorical_cross_entropy",
    "huber_loss",
    "accuracy",
    "precision",
    "recall",
    "f1_score",
    "confusion_matrix",
    "r2_score",
    "xavier_uniform",
    "xavier_normal",
    "he_uniform",
    "he_normal",
    "Value",
    "SGD",
    "Momentum",
    "Adam",
]
