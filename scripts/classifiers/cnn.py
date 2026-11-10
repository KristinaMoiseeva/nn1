import numpy as np
from ..layers import affine_forward, affine_backward, relu_forward, relu_backward, softmax_loss
from ..fast_layers import conv_forward_fast, conv_backward_fast, max_pool_forward_fast, max_pool_backward_fast


class ThreeLayerConvNet:
    def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
                 hidden_dim=100, num_classes=10, weight_scale=1e-3,
                 reg=0.0, dtype=np.float32):
        c, h, w = input_dim
        pooled_h, pooled_w = h // 2, w // 2
        self.params = {
            "W1": weight_scale * np.random.randn(num_filters, c, filter_size, filter_size),
            "b1": np.zeros(num_filters),
            "W2": weight_scale * np.random.randn(num_filters * pooled_h * pooled_w, hidden_dim),
            "b2": np.zeros(hidden_dim),
            "W3": weight_scale * np.random.randn(hidden_dim, num_classes),
            "b3": np.zeros(num_classes),
        }
        self.reg = reg
        self.dtype = dtype
        for k in self.params:
            self.params[k] = self.params[k].astype(dtype)

    def loss(self, X, y=None):
        X = X.astype(self.dtype)
        W1, b1 = self.params["W1"], self.params["b1"]
        W2, b2 = self.params["W2"], self.params["b2"]
        W3, b3 = self.params["W3"], self.params["b3"]
        conv_param = {"stride": 1, "pad": (W1.shape[2] - 1) // 2}
        pool_param = {"pool_height": 2, "pool_width": 2, "stride": 2}

        a1, conv_cache = conv_forward_fast(X, W1, b1, conv_param)
        r1, relu1_cache = relu_forward(a1)
        p1, pool_cache = max_pool_forward_fast(r1, pool_param)
        a2, fc2_cache = affine_forward(p1, W2, b2)
        r2, relu2_cache = relu_forward(a2)
        scores, fc3_cache = affine_forward(r2, W3, b3)
        if y is None:
            return scores

        loss, dscores = softmax_loss(scores, y)
        loss += 0.5 * self.reg * (np.sum(W1 ** 2) + np.sum(W2 ** 2) + np.sum(W3 ** 2))
        dr2, dW3, db3 = affine_backward(dscores, fc3_cache)
        da2 = relu_backward(dr2, relu2_cache)
        dp1, dW2, db2 = affine_backward(da2, fc2_cache)
        dr1 = max_pool_backward_fast(dp1, pool_cache)
        da1 = relu_backward(dr1, relu1_cache)
        _, dW1, db1 = conv_backward_fast(da1, conv_cache)
        return loss, {
            "W1": dW1 + self.reg * W1, "b1": db1,
            "W2": dW2 + self.reg * W2, "b2": db2,
            "W3": dW3 + self.reg * W3, "b3": db3,
        }
