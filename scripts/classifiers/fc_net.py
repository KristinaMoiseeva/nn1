import numpy as np
from ..layers import (
    affine_forward, affine_backward, relu_forward, relu_backward,
    batchnorm_forward, batchnorm_backward, dropout_forward, dropout_backward,
    softmax_loss,
)


class TwoLayerNet:
    def __init__(self, input_dim=3 * 32 * 32, hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0):
        self.params = {
            "W1": weight_scale * np.random.randn(input_dim, hidden_dim),
            "b1": np.zeros(hidden_dim),
            "W2": weight_scale * np.random.randn(hidden_dim, num_classes),
            "b2": np.zeros(num_classes),
        }
        self.reg = reg

    def loss(self, X, y=None):
        hidden, cache1 = affine_forward(X, self.params["W1"], self.params["b1"])
        hidden, relu_cache = relu_forward(hidden)
        scores, cache2 = affine_forward(hidden, self.params["W2"], self.params["b2"])
        if y is None:
            return scores
        loss, dscores = softmax_loss(scores, y)
        loss += 0.5 * self.reg * (np.sum(self.params["W1"] ** 2) + np.sum(self.params["W2"] ** 2))
        dhidden, dW2, db2 = affine_backward(dscores, cache2)
        dhidden = relu_backward(dhidden, relu_cache)
        _, dW1, db1 = affine_backward(dhidden, cache1)
        return loss, {
            "W1": dW1 + self.reg * self.params["W1"], "b1": db1,
            "W2": dW2 + self.reg * self.params["W2"], "b2": db2,
        }


class FullyConnectedNet:
    def __init__(self, hidden_dims, input_dim=3 * 32 * 32, num_classes=10, dropout=1,
                 normalization=None, reg=0.0, weight_scale=1e-2,
                 dtype=np.float32, seed=None):
        self.normalization = normalization
        self.use_dropout = dropout != 1
        self.reg = reg
        self.num_layers = len(hidden_dims) + 1
        self.dtype = dtype
        self.params = {}
        dims = [input_dim] + list(hidden_dims) + [num_classes]
        for i in range(1, self.num_layers + 1):
            self.params[f"W{i}"] = (weight_scale * np.random.randn(dims[i - 1], dims[i])).astype(dtype)
            self.params[f"b{i}"] = np.zeros(dims[i], dtype=dtype)
            if normalization == "batchnorm" and i < self.num_layers:
                self.params[f"gamma{i}"] = np.ones(dims[i], dtype=dtype)
                self.params[f"beta{i}"] = np.zeros(dims[i], dtype=dtype)
        self.dropout_param = {"mode": "train", "p": dropout}
        if seed is not None:
            self.dropout_param["seed"] = seed
        self.bn_params = [{"mode": "train"} for _ in hidden_dims] if normalization == "batchnorm" else []

    def loss(self, X, y=None):
        X = X.astype(self.dtype)
        mode = "test" if y is None else "train"
        if self.use_dropout:
            self.dropout_param["mode"] = mode
        if self.normalization == "batchnorm":
            for p in self.bn_params:
                p["mode"] = mode

        caches = []
        out = X
        for i in range(1, self.num_layers):
            out, fc_cache = affine_forward(out, self.params[f"W{i}"], self.params[f"b{i}"])
            bn_cache = None
            if self.normalization == "batchnorm":
                out, bn_cache = batchnorm_forward(
                    out, self.params[f"gamma{i}"], self.params[f"beta{i}"], self.bn_params[i - 1]
                )
            out, relu_cache = relu_forward(out)
            drop_cache = None
            if self.use_dropout:
                out, drop_cache = dropout_forward(out, self.dropout_param)
            caches.append((fc_cache, bn_cache, relu_cache, drop_cache))

        scores, final_cache = affine_forward(out, self.params[f"W{self.num_layers}"], self.params[f"b{self.num_layers}"])
        if y is None:
            return scores

        loss, dout = softmax_loss(scores, y)
        for i in range(1, self.num_layers + 1):
            loss += 0.5 * self.reg * np.sum(self.params[f"W{i}"] ** 2)

        grads = {}
        dout, dW, db = affine_backward(dout, final_cache)
        grads[f"W{self.num_layers}"] = dW + self.reg * self.params[f"W{self.num_layers}"]
        grads[f"b{self.num_layers}"] = db

        for i in reversed(range(1, self.num_layers)):
            fc_cache, bn_cache, relu_cache, drop_cache = caches[i - 1]
            if self.use_dropout:
                dout = dropout_backward(dout, drop_cache)
            dout = relu_backward(dout, relu_cache)
            if self.normalization == "batchnorm":
                dout, dgamma, dbeta = batchnorm_backward(dout, bn_cache)
                grads[f"gamma{i}"] = dgamma
                grads[f"beta{i}"] = dbeta
            dout, dW, db = affine_backward(dout, fc_cache)
            grads[f"W{i}"] = dW + self.reg * self.params[f"W{i}"]
            grads[f"b{i}"] = db
        return loss, grads
