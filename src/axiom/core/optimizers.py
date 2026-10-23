"""Parameter-update rules.

Both optimizers update a list of NumPy arrays in place from a matching list of
gradients. They are standalone for now: Dense still applies its own momentum
update inside backward().
"""
import numpy as np


def _check(params, grads):
    if len(params) != len(grads):
        raise ValueError(f"got {len(params)} params but {len(grads)} grads")
    for p, g in zip(params, grads):
        if not isinstance(p, np.ndarray):
            raise TypeError("params must be NumPy arrays, which are updated in place")
        if np.shape(g) != p.shape:
            raise ValueError(f"a gradient of shape {np.shape(g)} does not match its param of shape {p.shape}")


class SGD:
    """Vanilla gradient descent:  W := W - lr * dJ/dW"""

    def __init__(self, lr=0.01):
        self.lr = lr

    def step(self, params: list[np.ndarray], grads: list[np.ndarray]) -> None:
        _check(params, grads)
        for p, g in zip(params, grads):
            p -= self.lr * g


class MomentumSGD:
    """Gradient descent on an exponentially weighted average of the gradients:

        v := beta * v + (1 - beta) * dJ/dW
        W := W - lr * v

    `momentum` is accepted as another name for `beta`. One velocity is kept per
    position in the params list, so pass the params in the same order each step.
    """

    def __init__(self, lr=0.01, beta=0.9, *, momentum=None):
        if momentum is not None:
            beta = momentum
        if not 0.0 <= beta < 1.0:
            raise ValueError(f"beta must be in [0, 1), got {beta}")
        self.lr = lr
        self.beta = beta
        self.velocities = None

    @property
    def momentum(self):
        return self.beta

    def step(self, params: list[np.ndarray], grads: list[np.ndarray]) -> None:
        _check(params, grads)
        if self.velocities is None or [v.shape for v in self.velocities] != [p.shape for p in params]:
            self.velocities = [np.zeros_like(p, dtype=float) for p in params]
        for p, g, v in zip(params, grads, self.velocities):
            v *= self.beta
            v += (1.0 - self.beta) * g
            p -= self.lr * v
