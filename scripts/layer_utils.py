from .layers import affine_forward, affine_backward, relu_forward, relu_backward
from .fast_layers import conv_forward_fast, conv_backward_fast, max_pool_forward_fast, max_pool_backward_fast


def affine_relu_forward(x, w, b):
    a, fc_cache = affine_forward(x, w, b)
    out, relu_cache = relu_forward(a)
    return out, (fc_cache, relu_cache)


def affine_relu_backward(dout, cache):
    fc_cache, relu_cache = cache
    da = relu_backward(dout, relu_cache)
    return affine_backward(da, fc_cache)


def conv_relu_forward(x, w, b, conv_param):
    a, conv_cache = conv_forward_fast(x, w, b, conv_param)
    out, relu_cache = relu_forward(a)
    return out, (conv_cache, relu_cache)


def conv_relu_backward(dout, cache):
    conv_cache, relu_cache = cache
    da = relu_backward(dout, relu_cache)
    return conv_backward_fast(da, conv_cache)


def conv_relu_pool_forward(x, w, b, conv_param, pool_param):
    a, conv_cache = conv_forward_fast(x, w, b, conv_param)
    s, relu_cache = relu_forward(a)
    out, pool_cache = max_pool_forward_fast(s, pool_param)
    return out, (conv_cache, relu_cache, pool_cache)


def conv_relu_pool_backward(dout, cache):
    conv_cache, relu_cache, pool_cache = cache
    ds = max_pool_backward_fast(dout, pool_cache)
    da = relu_backward(ds, relu_cache)
    return conv_backward_fast(da, conv_cache)
