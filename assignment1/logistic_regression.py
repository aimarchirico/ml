import numpy as np

class LogisticRegression:
    def __init__(self, learning_rate=0.01, n_iters=1000, tolerance=1e-8, fit_intercept=True, random_state=None):
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.tolerance = tolerance
        self.fit_intercept = fit_intercept
        if random_state is not None:
            np.random.seed(random_state)
        
        # Learned parameters
        self.coef_ = None
        self.intercept_ = 0.0
        self.fitted_ = False
    
    def _prepare_X(self, X):
        """Convert input to proper format"""
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return X
    
    def _sigmoid(self, z):
        """Sigmoid activation function with numerical stability"""
        # Clip z to prevent overflow
        z = np.clip(z, -250, 250)
        return 1 / (1 + np.exp(-z))
    
    def fit(self, X, y):
        """
        Fit logistic regression using gradient descent
        
        Args:
            X (array): feature matrix
            y (array): binary target vector (0s and 1s)
        """
        X = self._prepare_X(X)
        y = np.asarray(y, dtype=float).ravel()
        
        m, n = X.shape
        if y.shape[0] != m:
            raise ValueError("X and y have incompatible shapes")
        
        # Initialize parameters
        self.coef_ = np.random.normal(0, 0.01, n)  # Small random initialization
        self.intercept_ = 0.0
        
        prev_loss = np.inf
        
        for i in range(self.n_iters):
            # Forward pass
            z = X @ self.coef_ + (self.intercept_ if self.fit_intercept else 0.0)
            y_pred = self._sigmoid(z)
            
            # Compute loss (cross-entropy)
            # Add small epsilon to prevent log(0)
            epsilon = 1e-15
            y_pred_clipped = np.clip(y_pred, epsilon, 1 - epsilon)
            loss = -(1/m) * np.sum(y * np.log(y_pred_clipped) + (1 - y) * np.log(1 - y_pred_clipped))
            
            # Compute gradients
            errors = y_pred - y
            grad_w = (1/m) * (X.T @ errors)
            if self.fit_intercept:
                grad_b = (1/m) * np.sum(errors)
            
            # Update parameters
            self.coef_ -= self.learning_rate * grad_w
            if self.fit_intercept:
                self.intercept_ -= self.learning_rate * grad_b
            
            # Early stopping
            if self.tolerance is not None:
                if prev_loss - loss >= 0 and (prev_loss - loss) < self.tolerance:
                    break
            prev_loss = loss
        
        self.fitted_ = True
        return self
    
    def predict_proba(self, X):
        """Predict class probabilities"""
        if not self.fitted_:
            raise RuntimeError("Model must be fitted before calling predict_proba()")
        
        X = self._prepare_X(X)
        if X.shape[1] != self.coef_.shape[0]:
            raise ValueError("Input has different number of features than training data")
        
        z = X @ self.coef_ + (self.intercept_ if self.fit_intercept else 0.0)
        return self._sigmoid(z)
    
    def predict(self, X):
        """Predict binary classes"""
        probabilities = self.predict_proba(X)
        return (probabilities >= 0.5).astype(int)