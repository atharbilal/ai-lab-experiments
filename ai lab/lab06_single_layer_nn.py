"""
LAB 06: Single-Layer Neural Network (Perceptron) for Classification
Implemented from scratch with NumPy.

  y_hat = step(w . x + b)
  Update rule:  w <- w + lr * (y - y_hat) * x
                b <- b + lr * (y - y_hat)
"""
import numpy as np


class Perceptron:
    def __init__(self, lr=0.1, epochs=100, seed=42):
        self.lr = lr
        self.epochs = epochs
        self.rng = np.random.default_rng(seed)
        self.w = None
        self.b = 0.0
        self.errors_per_epoch = []

    @staticmethod
    def activation(z):
        return np.where(z >= 0, 1, 0)        #unit step fxn 

    def net_input(self, X):
        return X @ self.w + self.b

    def predict(self, X):
        return self.activation(self.net_input(X))

    def fit(self, X, y):
        n_features = X.shape[1]
        self.w = self.rng.normal(0, 0.01, n_features)
        self.b = 0.0
        self.errors_per_epoch = []
        for _ in range(self.epochs):
            errors = 0
            for xi, target in zip(X, y):
                update = self.lr * (target - self.predict(xi.reshape(1, -1))[0])
                self.w += update * xi
                self.b += update
                errors += int(update != 0)
            self.errors_per_epoch.append(errors)
            if errors == 0:                  # converged
                break
        return self

    def accuracy(self, X, y):
        return np.mean(self.predict(X) == y)


def demo_logic_gates():
    print("=" * 55)
    print("Part 1: Logic gates")
    print("=" * 55)
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    gates = {"AND": np.array([0, 0, 0, 1]),
             "OR":  np.array([0, 1, 1, 1]),
             "XOR": np.array([0, 1, 1, 0])}
    for name, y in gates.items():
        p = Perceptron(lr=0.1, epochs=50).fit(X, y)
        preds = p.predict(X)
        print(f"{name:<4} predictions={preds.tolist()}  target={y.tolist()}  "
              f"accuracy={p.accuracy(X, y):.2f}  epochs={len(p.errors_per_epoch)}")
    print("-> XOR is not linearly separable, so a single-layer network cannot learn it.")


def make_dataset(n=200, seed=0):
    """Two Gaussian blobs (linearly separable) in 2D."""
    rng = np.random.default_rng(seed)
    class0 = rng.normal(loc=[2, 2], scale=0.8, size=(n // 2, 2))
    class1 = rng.normal(loc=[6, 6], scale=0.8, size=(n // 2, 2))
    X = np.vstack([class0, class1])
    y = np.array([0] * (n // 2) + [1] * (n // 2))
    idx = rng.permutation(n)
    return X[idx], y[idx]


def demo_classification():
    print("\n" + "=" * 55)
    print("Part 2: Binary classification on 2D data")
    print("=" * 55)
    X, y = make_dataset()
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # standardize using training statistics
    mu, sigma = X_train.mean(axis=0), X_train.std(axis=0)
    X_train = (X_train - mu) / sigma
    X_test = (X_test - mu) / sigma

    model = Perceptron(lr=0.01, epochs=100).fit(X_train, y_train)
    print(f"Learned weights: {np.round(model.w, 4)}, bias: {model.b:.4f}")
    print(f"Converged after {len(model.errors_per_epoch)} epochs; "
          f"errors per epoch: {model.errors_per_epoch}")
    print(f"Train accuracy: {model.accuracy(X_train, y_train) * 100:.2f}%")
    print(f"Test accuracy:  {model.accuracy(X_test, y_test) * 100:.2f}%")

    # confusion matrix
    pred = model.predict(X_test)
    tp = int(np.sum((pred == 1) & (y_test == 1)))
    tn = int(np.sum((pred == 0) & (y_test == 0)))
    fp = int(np.sum((pred == 1) & (y_test == 0)))
    fn = int(np.sum((pred == 0) & (y_test == 1)))
    print(f"Confusion matrix (test):  TN={tn} FP={fp} | FN={fn} TP={tp}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
        Xs = np.vstack([X_train, X_test])
        ys = np.concatenate([y_train, y_test])
        ax[0].scatter(Xs[ys == 0, 0], Xs[ys == 0, 1], c="tab:blue", label="Class 0", s=18)
        ax[0].scatter(Xs[ys == 1, 0], Xs[ys == 1, 1], c="tab:orange", label="Class 1", s=18)
        xs = np.linspace(Xs[:, 0].min() - 0.5, Xs[:, 0].max() + 0.5, 100)
        ax[0].plot(xs, -(model.w[0] * xs + model.b) / model.w[1], "k--", label="Decision boundary")
        ax[0].set_title("Perceptron decision boundary (standardized)")
        ax[0].legend()
        ax[1].plot(range(1, len(model.errors_per_epoch) + 1), model.errors_per_epoch, marker="o")
        ax[1].set_title("Misclassifications per epoch")
        ax[1].set_xlabel("Epoch")
        ax[1].set_ylabel("Errors")
        plt.tight_layout()
        plt.savefig("lab06_perceptron.png", dpi=120)
        print("Plot saved to lab06_perceptron.png")
    except ImportError:
        print("(matplotlib not installed - skipping plot)")


if __name__ == "__main__":
    demo_logic_gates()
    demo_classification()
