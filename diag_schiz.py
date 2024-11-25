import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

# Load the data
train_fnc = pd.read_csv('./data/Train/train_FNC.csv')
train_sbm = pd.read_csv('./data/Train/train_SBM.csv')
train_labels = pd.read_csv('./data/Train/train_labels.csv')

# Merge features and labels
train_data = pd.merge(train_fnc, train_sbm, on='Id')  # Merge FNC and SBM data on the 'Id' column
train_data = pd.merge(train_data, train_labels, on='Id')  # Add labels

# Separate features (X) and labels (y)
X = train_data.drop(columns=['Id', 'Class'])  # Drop 'Id' and 'Class' (target) from features
y = train_data['Class']  # 'Class' is the target column

# Handle missing values
X.fillna(X.mean(), inplace=True)

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train Logistic Regression
lr_model = LogisticRegression(random_state=42, max_iter=500)
lr_model.fit(X_train, y_train)

# Evaluate the model
y_pred = lr_model.predict(X_test)
y_proba = lr_model.predict_proba(X_test)[:, 1]  # Probabilities for ROC-AUC

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)

print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"ROC-AUC: {roc_auc:.2f}")
