import numpy as np


def sgd(w, dw, config=None):
    config = {} if config is None else config
    lr = config.setdefault("learning_rate", 1e-2)
    return w - lr * dw, config


def sgd_momentum(w, dw, config=None):
    config = {} if config is None else config
    lr = config.setdefault("learning_rate", 1e-2)
    momentum = config.setdefault("momentum", 0.9)
    velocity = config.get("velocity", np.zeros_like(w))
    velocity = momentum * velocity - lr * dw
    next_w = w + velocity
    config["velocity"] = velocity
    return next_w, config


def rmsprop(w, dw, config=None):
    config = {} if config is None else config
    lr = config.setdefault("learning_rate", 1e-2)
    decay = config.setdefault("decay_rate", 0.99)
    eps = config.setdefault("epsilon", 1e-8)
    cache = config.get("cache", np.zeros_like(w))
    cache = decay * cache + (1 - decay) * (dw ** 2)
    next_w = w - lr * dw / (np.sqrt(cache) + eps)
    config["cache"] = cache
    return next_w, config


def adam(w, dw, config=None):
    config = {} if config is None else config
    lr = config.setdefault("learning_rate", 1e-3)
    beta1 = config.setdefault("beta1", 0.9)
    beta2 = config.setdefault("beta2", 0.999)
    eps = config.setdefault("epsilon", 1e-8)
    m = config.get("m", np.zeros_like(w))
    v = config.get("v", np.zeros_like(w))
    t = config.get("t", 0) + 1
    m = beta1 * m + (1 - beta1) * dw
    v = beta2 * v + (1 - beta2) * (dw ** 2)
    m_hat = m / (1 - beta1 ** t)
    v_hat = v / (1 - beta2 ** t)
    next_w = w - lr * m_hat / (np.sqrt(v_hat) + eps)
    config.update({"m": m, "v": v, "t": t})
    return next_w, config
