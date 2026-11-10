import numpy as np

try:
    from .im2col_cython import im2col_cython, col2im_cython
except ImportError:
    im2col_cython = col2im_cython = None


def _im2col(x, hh, ww, pad, stride):
    n, c, h, w = x.shape
    if im2col_cython is not None:
        return im2col_cython(x, hh, ww, pad, stride), (n, c, h + 2 * pad, w + 2 * pad)
    out_h = 1 + (h + 2 * pad - hh) // stride
    out_w = 1 + (w + 2 * pad - ww) // stride
    x_pad = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
    shape = (n, c, hh, ww, out_h, out_w)
    strides = (
        x_pad.strides[0], x_pad.strides[1], x_pad.strides[2], x_pad.strides[3],
        stride * x_pad.strides[2], stride * x_pad.strides[3],
    )
    windows = np.lib.stride_tricks.as_strided(x_pad, shape=shape, strides=strides, writeable=False)
    cols = windows.transpose(1, 2, 3, 0, 4, 5).reshape(c * hh * ww, -1)
    return cols, x_pad.shape


def _col2im(cols, x_shape, hh, ww, pad, stride):
    n, c, h, w = x_shape
    if col2im_cython is not None:
        return col2im_cython(cols, n, c, h, w, hh, ww, pad, stride)
    out_h = 1 + (h + 2 * pad - hh) // stride
    out_w = 1 + (w + 2 * pad - ww) // stride
    dx_pad = np.zeros((n, c, h + 2 * pad, w + 2 * pad), dtype=cols.dtype)
    reshaped = cols.reshape(c, hh, ww, n, out_h, out_w).transpose(3, 0, 1, 2, 4, 5)
    for r in range(hh):
        r_max = r + stride * out_h
        for col in range(ww):
            c_max = col + stride * out_w
            dx_pad[:, :, r:r_max:stride, col:c_max:stride] += reshaped[:, :, r, col]
    return dx_pad if pad == 0 else dx_pad[:, :, pad:-pad, pad:-pad]


def conv_forward_fast(x, w, b, conv_param):
    stride, pad = conv_param["stride"], conv_param["pad"]
    f, _, hh, ww = w.shape
    cols, _ = _im2col(x, hh, ww, pad, stride)
    out = w.reshape(f, -1) @ cols + b[:, None]
    out_h = 1 + (x.shape[2] + 2 * pad - hh) // stride
    out_w = 1 + (x.shape[3] + 2 * pad - ww) // stride
    out = out.reshape(f, x.shape[0], out_h, out_w).transpose(1, 0, 2, 3)
    return out, (x, w, b, conv_param, cols)


def conv_backward_fast(dout, cache):
    x, w, _, conv_param, cols = cache
    stride, pad = conv_param["stride"], conv_param["pad"]
    f, _, hh, ww = w.shape
    dout_flat = dout.transpose(1, 0, 2, 3).reshape(f, -1)
    db = dout_flat.sum(axis=1)
    dw = (dout_flat @ cols.T).reshape(w.shape)
    dcols = w.reshape(f, -1).T @ dout_flat
    dx = _col2im(dcols, x.shape, hh, ww, pad, stride)
    return dx, dw, db


def max_pool_forward_fast(x, pool_param):
    ph, pw, stride = pool_param["pool_height"], pool_param["pool_width"], pool_param["stride"]
    n, c, h, w = x.shape
    x_reshaped = x.reshape(n * c, 1, h, w)
    cols, _ = _im2col(x_reshaped, ph, pw, 0, stride)
    max_idx = np.argmax(cols, axis=0)
    out = cols[max_idx, np.arange(cols.shape[1])]
    out_h = 1 + (h - ph) // stride
    out_w = 1 + (w - pw) // stride
    out = out.reshape(n, c, out_h, out_w)
    return out, (x, pool_param, cols, max_idx)


def max_pool_backward_fast(dout, cache):
    x, pool_param, cols, max_idx = cache
    ph, pw, stride = pool_param["pool_height"], pool_param["pool_width"], pool_param["stride"]
    n, c, h, w = x.shape
    dcols = np.zeros_like(cols)
    dout_flat = dout.reshape(-1)
    dcols[max_idx, np.arange(dcols.shape[1])] = dout_flat
    dx = _col2im(dcols, (n * c, 1, h, w), ph, pw, 0, stride)
    return dx.reshape(x.shape)
