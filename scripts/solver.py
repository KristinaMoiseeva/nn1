import copy
import numpy as np
from . import optim


class Solver:
    def __init__(self, model, data, update_rule="sgd", optim_config=None,
                 lr_decay=1.0, batch_size=100, num_epochs=10,
                 print_every=100, verbose=False, restore_best=True):
        self.model = model
        self.X_train = data["X_train"]
        self.y_train = data["y_train"]
        self.X_val = data["X_val"]
        self.y_val = data["y_val"]
        self.update_rule = getattr(optim, update_rule)
        self.optim_config = {} if optim_config is None else optim_config
        self.lr_decay = lr_decay
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.print_every = print_every
        self.verbose = verbose
        self.restore_best = restore_best
        self.best_bn_params = []
        self.loss_history = []
        self.train_acc_history = []
        self.val_acc_history = []
        self._optim_configs = {k: dict(self.optim_config) for k in model.params}
        self.best_val_acc = -1
        self.best_params = {}
        self.rng = np.random.default_rng(42)

    def _step(self):
        idx = self.rng.choice(self.X_train.shape[0], size=min(self.batch_size, self.X_train.shape[0]), replace=True)
        X_batch, y_batch = self.X_train[idx], self.y_train[idx]
        loss, grads = self.model.loss(X_batch, y_batch)
        self.loss_history.append(float(loss))
        for p, w in self.model.params.items():
            next_w, config = self.update_rule(w, grads[p], self._optim_configs[p])
            self.model.params[p] = next_w
            self._optim_configs[p] = config

    def check_accuracy(self, X, y, batch_size=256):
        preds = []
        for start in range(0, X.shape[0], batch_size):
            scores = self.model.loss(X[start:start + batch_size])
            preds.append(np.argmax(scores, axis=1))
        return np.mean(np.concatenate(preds) == y)

    def train(self):
        iterations_per_epoch = max(self.X_train.shape[0] // self.batch_size, 1)
        num_iterations = self.num_epochs * iterations_per_epoch
        for t in range(num_iterations):
            self._step()
            if self.verbose and (t % self.print_every == 0 or t == num_iterations - 1):
                print(f"Итерация {t + 1}/{num_iterations}: loss={self.loss_history[-1]:.5f}")
            if (t + 1) % iterations_per_epoch == 0:
                train_acc = self.check_accuracy(self.X_train, self.y_train)
                val_acc = self.check_accuracy(self.X_val, self.y_val)
                self.train_acc_history.append(float(train_acc))
                self.val_acc_history.append(float(val_acc))
                for k in self._optim_configs:
                    if "learning_rate" in self._optim_configs[k]:
                        self._optim_configs[k]["learning_rate"] *= self.lr_decay
                if val_acc > self.best_val_acc:
                    self.best_val_acc = val_acc
                    self.best_params = {k: v.copy() for k, v in self.model.params.items()}
                    self.best_bn_params = copy.deepcopy(getattr(self.model, "bn_params", []))
        if self.best_params and self.restore_best:
            self.model.params = self.best_params
            if hasattr(self.model, "bn_params"):
                self.model.bn_params = copy.deepcopy(self.best_bn_params)
