import numpy as np

class LinearRegression():
    def __init__(self, learning_rate: float = 0.01, n_iters: int = 1000, tolerance: float | None = 1e-8, random_state: int | None = None):
        # Hyperparameters
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.tolerance = tolerance
        if random_state is not None:
            np.random.seed(random_state)
        # Learned parameters
        self.coef_ = None  # shape (n_features,)
        self.intercept_ = 0.0
        self.fitted_ = False

    def _prepare_X(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return X

    def fit(self, X, y):
        """
        Estimates parameters for linear regression using batch gradient descent.

        Args:
            X (array<m,n>): feature matrix
            y (array<m>): target vector
        """
        X = self._prepare_X(X)
        y = np.asarray(y, dtype=float).ravel()
        m, n = X.shape
        if y.shape[0] != m:
            raise ValueError("X and y have incompatible shapes.")

        # Initialize weights (small random or zeros)
        self.coef_ = np.zeros(n)
        self.intercept_ = 0.0

        prev_loss = np.inf
        for i in range(self.n_iters):
            # Predictions
            y_pred = X @ self.coef_ + self.intercept_
            # Residuals
            errors = y_pred - y
            # Compute gradients
            grad_w = (1 / m) * (X.T @ errors)
            grad_b = (1 / m) * np.sum(errors)
            # Update parameters
            self.coef_ -= self.learning_rate * grad_w
            self.intercept_ -= self.learning_rate * grad_b
            # Early stopping based on loss improvement
            loss = (1 / (2 * m)) * np.sum(errors ** 2)  # MSE/2
            if self.tolerance is not None:
                if prev_loss - loss >= 0 and (prev_loss - loss) < self.tolerance:
                    break
            prev_loss = loss

        self.fitted_ = True
        return self

    def predict(self, X):
        """
        Generates predictions for input features.

        Args:
            X (array<m,n>): feature matrix
        Returns:
            array<m>: predictions
        """
        if not self.fitted_:
            raise RuntimeError("Model must be fitted before calling predict().")
        X = self._prepare_X(X)
        if X.shape[1] != self.coef_.shape[0]:
            raise ValueError("Input has different number of features than the model was trained on.")
        return X @ self.coef_ + self.intercept_





