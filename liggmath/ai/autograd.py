"""Minimal scalar-valued reverse-mode autograd engine.

Useful for teaching/prototyping backpropagation without pulling in a full
deep-learning framework. Build an expression out of `Value` nodes, call
`.backward()` on the final (scalar) result, and read `.grad` off any leaf.
"""

from __future__ import annotations

import math
from typing import Callable, Iterable


class Value:
    """A scalar value that tracks the graph of operations used to produce it."""

    __slots__ = ("data", "grad", "_backward", "_prev", "_op")

    def __init__(self, data: float, _children: Iterable["Value"] = (), _op: str = ""):
        self.data = float(data)
        self.grad = 0.0
        self._backward: Callable[[], None] = lambda: None
        self._prev = set(_children)
        self._op = _op

    def __repr__(self) -> str:
        return f"Value(data={self.data}, grad={self.grad})"

    @staticmethod
    def _wrap(x) -> "Value":
        return x if isinstance(x, Value) else Value(x)

    def __add__(self, other) -> "Value":
        other = Value._wrap(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    def __mul__(self, other) -> "Value":
        other = Value._wrap(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __pow__(self, power: float) -> "Value":
        assert isinstance(power, (int, float)), "only numeric powers are supported"
        out = Value(self.data**power, (self,), f"**{power}")

        def _backward():
            self.grad += (power * self.data ** (power - 1)) * out.grad

        out._backward = _backward
        return out

    def relu(self) -> "Value":
        out = Value(max(0.0, self.data), (self,), "relu")

        def _backward():
            self.grad += (out.data > 0) * out.grad

        out._backward = _backward
        return out

    def tanh(self) -> "Value":
        t = math.tanh(self.data)
        out = Value(t, (self,), "tanh")

        def _backward():
            self.grad += (1 - t**2) * out.grad

        out._backward = _backward
        return out

    def exp(self) -> "Value":
        e = math.exp(self.data)
        out = Value(e, (self,), "exp")

        def _backward():
            self.grad += e * out.grad

        out._backward = _backward
        return out

    def log(self) -> "Value":
        out = Value(math.log(self.data), (self,), "log")

        def _backward():
            self.grad += (1.0 / self.data) * out.grad

        out._backward = _backward
        return out

    def sigmoid(self) -> "Value":
        s = 1.0 / (1.0 + math.exp(-self.data))
        out = Value(s, (self,), "sigmoid")

        def _backward():
            self.grad += s * (1 - s) * out.grad

        out._backward = _backward
        return out

    def backward(self) -> None:
        topo: list[Value] = []
        visited: set[Value] = set()

        def build(v: "Value"):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build(child)
                topo.append(v)

        build(self)
        self.grad = 1.0
        for v in reversed(topo):
            v._backward()

    def zero_grad(self) -> None:
        self.grad = 0.0

    def __neg__(self):
        return self * -1

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return self + (-Value._wrap(other))

    def __rsub__(self, other):
        return Value._wrap(other) + (-self)

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        return self * Value._wrap(other) ** -1

    def __rtruediv__(self, other):
        return Value._wrap(other) * self**-1
