import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, log_loss
from logistic_regression import LogisticRegression  # Import the custom Logistic Regression class

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

# Save the trained model and scaler
with open('./project/model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('./project/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("Custom model and scaler saved successfully!")