import numpy as np


def affine_forward(x, w, b):
    x_row = x.reshape(x.shape[0], -1)
    out = x_row @ w + b
    return out, (x, w, b)


def affine_backward(dout, cache):
    x, w, _ = cache
    x_row = x.reshape(x.shape[0], -1)
    dx = (dout @ w.T).reshape(x.shape)
    dw = x_row.T @ dout
    db = np.sum(dout, axis=0)
    return dx, dw, db


def relu_forward(x):
    return np.maximum(0, x), x


def relu_backward(dout, cache):
    return dout * (cache > 0)


def batchnorm_forward(x, gamma, beta, bn_param):
    mode = bn_param["mode"]
    eps = bn_param.get("eps", 1e-5)
    momentum = bn_param.get("momentum", 0.9)
    running_mean = bn_param.get("running_mean", np.zeros(x.shape[1], dtype=x.dtype))
    running_var = bn_param.get("running_var", np.zeros(x.shape[1], dtype=x.dtype))

    if mode == "train":
        mean = x.mean(axis=0)
        var = x.var(axis=0)
        inv_std = 1.0 / np.sqrt(var + eps)
        x_hat = (x - mean) * inv_std
        out = gamma * x_hat + beta
        running_mean = momentum * running_mean + (1 - momentum) * mean
        running_var = momentum * running_var + (1 - momentum) * var
        cache = (x_hat, gamma, inv_std)
    elif mode == "test":
        x_hat = (x - running_mean) / np.sqrt(running_var + eps)
        out = gamma * x_hat + beta
        cache = None
    else:
        raise ValueError(f"Неизвестный режим BatchNorm: {mode}")

    bn_param["running_mean"] = running_mean
    bn_param["running_var"] = running_var
    return out, cache


def batchnorm_backward(dout, cache):
    x_hat, gamma, inv_std = cache
    n = dout.shape[0]
    dbeta = dout.sum(axis=0)
    dgamma = np.sum(dout * x_hat, axis=0)
    dx = gamma * inv_std / n * (
        n * dout - dout.sum(axis=0) - x_hat * np.sum(dout * x_hat, axis=0)
    )
    return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
    return batchnorm_backward(dout, cache)


def dropout_forward(x, dropout_param):
    p = dropout_param["p"]
    mode = dropout_param["mode"]
    if not 0 < p <= 1:
        raise ValueError("p должен быть в диапазоне (0, 1]")
    mask = None
    if mode == "train":
        if "seed" in dropout_param:
            rng = np.random.default_rng(dropout_param["seed"])
        else:
            rng = np.random
        mask = ((rng.random(x.shape) < p) / p).astype(x.dtype)
        out = x * mask
    elif mode == "test":
        out = x
    else:
        raise ValueError(f"Неизвестный режим Dropout: {mode}")
    return out.astype(x.dtype, copy=False), (dropout_param.copy(), mask)


def dropout_backward(dout, cache):
    dropout_param, mask = cache
    if dropout_param["mode"] == "train":
        return dout * mask
    return dout


def conv_forward_naive(x, w, b, conv_param):
    stride, pad = conv_param["stride"], conv_param["pad"]
    n, c, h, width = x.shape
    f, _, hh, ww = w.shape
    out_h = 1 + (h + 2 * pad - hh) // stride
    out_w = 1 + (width + 2 * pad - ww) // stride
    x_pad = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
    out = np.zeros((n, f, out_h, out_w), dtype=x.dtype)
    for i in range(n):
        for filt in range(f):
            for r in range(out_h):
                hs = r * stride
                for col in range(out_w):
                    ws = col * stride
                    window = x_pad[i, :, hs:hs + hh, ws:ws + ww]
                    out[i, filt, r, col] = np.sum(window * w[filt]) + b[filt]
    return out, (x, w, b, conv_param)


def conv_backward_naive(dout, cache):
    x, w, _, conv_param = cache
    stride, pad = conv_param["stride"], conv_param["pad"]
    n, c, h, width = x.shape
    f, _, hh, ww = w.shape
    _, _, out_h, out_w = dout.shape
    x_pad = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
    dx_pad = np.zeros_like(x_pad)
    dw = np.zeros_like(w)
    db = dout.sum(axis=(0, 2, 3))
    for i in range(n):
        for filt in range(f):
            for r in range(out_h):
                hs = r * stride
                for col in range(out_w):
                    ws = col * stride
                    grad = dout[i, filt, r, col]
                    window = x_pad[i, :, hs:hs + hh, ws:ws + ww]
                    dw[filt] += window * grad
                    dx_pad[i, :, hs:hs + hh, ws:ws + ww] += w[filt] * grad
    dx = dx_pad if pad == 0 else dx_pad[:, :, pad:-pad, pad:-pad]
    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    ph, pw, stride = pool_param["pool_height"], pool_param["pool_width"], pool_param["stride"]
    n, c, h, w = x.shape
    out_h = 1 + (h - ph) // stride
    out_w = 1 + (w - pw) // stride
    out = np.zeros((n, c, out_h, out_w), dtype=x.dtype)
    for i in range(n):
        for ch in range(c):
            for r in range(out_h):
                hs = r * stride
                for col in range(out_w):
                    ws = col * stride
                    out[i, ch, r, col] = np.max(x[i, ch, hs:hs + ph, ws:ws + pw])
    return out, (x, pool_param)


def max_pool_backward_naive(dout, cache):
    x, pool_param = cache
    ph, pw, stride = pool_param["pool_height"], pool_param["pool_width"], pool_param["stride"]
    n, c, _, _ = x.shape
    _, _, out_h, out_w = dout.shape
    dx = np.zeros_like(x)
    for i in range(n):
        for ch in range(c):
            for r in range(out_h):
                hs = r * stride
                for col in range(out_w):
                    ws = col * stride
                    window = x[i, ch, hs:hs + ph, ws:ws + pw]
                    mask = np.zeros_like(window)
                    mask.flat[np.argmax(window)] = 1
                    dx[i, ch, hs:hs + ph, ws:ws + pw] += mask * dout[i, ch, r, col]
    return dx


def softmax_loss(x, y):
    shifted = x - np.max(x, axis=1, keepdims=True)
    exp_scores = np.exp(shifted)
    probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)
    n = x.shape[0]
    loss = np.mean(np.log(exp_scores.sum(axis=1)) - shifted[np.arange(n), y])
    dx = probs.copy()
    dx[np.arange(n), y] -= 1
    dx /= n
    return loss, dx
