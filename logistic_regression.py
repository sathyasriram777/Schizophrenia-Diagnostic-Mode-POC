import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, log_loss
import seaborn as sns

# Logistic function
def logistic(z):
    return 1.0 / (1 + np.exp(-z))

# Custom Logistic Regression class
class LogisticRegression:
    def __init__(self, learning_rate=0.01, max_iters=1000, epsilon=1e-5, 
                 l1_reg=0, l2_reg=0, add_bias=True, batch_size=None):
        self.learning_rate = learning_rate
        self.max_iters = max_iters
        self.epsilon = epsilon
        self.l1_reg = l1_reg
        self.l2_reg = l2_reg
        self.add_bias = add_bias
        self.batch_size = batch_size
        self.w = None

    def fit(self, x, y):
        if x.ndim == 1:
            x = x[:, None]

        # Add bias
        if self.add_bias:
            x = np.column_stack([x, np.ones(x.shape[0])])
        
        # Initialize weights
        n, d = x.shape
        self.w = np.zeros(d)

        # Gradient descent
        for _ in range(int(self.max_iters)):
            if self.batch_size:  # Mini-batch gradient descent
                indices = np.random.choice(n, self.batch_size, replace=False)
                x_batch = x[indices]
                y_batch = y[indices]
            else:
                x_batch, y_batch = x, y

            yh = logistic(np.dot(x_batch, self.w))
            error = yh - y_batch

            # Gradient calculation
            grad = np.dot(x_batch.T, error) / len(y_batch)
            if self.l1_reg:
                grad += self.l1_reg * np.sign(self.w)
            if self.l2_reg:
                grad += self.l2_reg * self.w
            
            # Update weights
            self.w -= self.learning_rate * grad
            
            # Check for convergence
            if np.linalg.norm(grad) < self.epsilon:
                break

    def predict_proba(self, x):
        if x.ndim == 1:
            x = x[:, None]
        if self.add_bias:
            x = np.column_stack([x, np.ones(x.shape[0])])
        return logistic(np.dot(x, self.w))

    def predict(self, x):
        return (self.predict_proba(x) >= 0.5).astype(int)

# Data Preparation
train_fnc = pd.read_csv('./data/Train/train_FNC.csv')
train_sbm = pd.read_csv('./data/Train/train_SBM.csv')
train_labels = pd.read_csv('./data/Train/train_labels.csv')

# Merge features and labels
train_data = pd.merge(train_fnc, train_sbm, on='Id')
train_data = pd.merge(train_data, train_labels, on='Id')

# Prepare features and labels
X = train_data.drop(columns=['Id', 'Class']).values
y = train_data['Class'].values

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split Data
def split_data(X, y, split_ratio=0.8):
    split_idx = int(len(X) * split_ratio)
    return X[:split_idx], X[split_idx:], y[:split_idx], y[split_idx:]

X_train, X_test, y_train, y_test = split_data(X_scaled, y)

# Train Custom Logistic Regression
model = LogisticRegression(learning_rate=0.01, max_iters=1000, l1_reg=0.01, l2_reg=0.01, batch_size=32)
model.fit(X_train, y_train)

# Predictions
train_preds = model.predict(X_train)
test_preds = model.predict(X_test)
train_probs = model.predict_proba(X_train)
test_probs = model.predict_proba(X_test)

# Performance Metrics
def performance_metrics(y_true, y_pred, y_probs):
    acc = np.mean(y_true == y_pred)
    auc = roc_auc_score(y_true, y_probs)
    logloss = log_loss(y_true, y_probs)
    print(f"Accuracy: {acc:.4f}")
    print(f"AUC: {auc:.4f}")
    print(f"Log Loss: {logloss:.4f}")

performance_metrics(y_test, test_preds, test_probs)
