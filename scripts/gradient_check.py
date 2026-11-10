import numpy as np


def eval_numerical_gradient(f, x, verbose=True, h=1e-5):
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"], op_flags=["readwrite"])
    while not it.finished:
        idx = it.multi_index
        old = x[idx]
        x[idx] = old + h
        pos = f(x)
        x[idx] = old - h
        neg = f(x)
        x[idx] = old
        grad[idx] = (pos - neg) / (2 * h)
        if verbose:
            print(idx, grad[idx])
        it.iternext()
    return grad


def eval_numerical_gradient_array(f, x, df, h=1e-5):
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"], op_flags=["readwrite"])
    while not it.finished:
        idx = it.multi_index
        old = x[idx]
        x[idx] = old + h
        pos = f(x).copy()
        x[idx] = old - h
        neg = f(x).copy()
        x[idx] = old
        grad[idx] = np.sum((pos - neg) * df) / (2 * h)
        it.iternext()
    return grad
