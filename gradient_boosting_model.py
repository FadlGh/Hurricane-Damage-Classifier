import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import pickle

X = []
y = []

for i in range(16): # Edit number based on how many chunks you have
    with open(f'X_chunk_{i}.pickle', 'rb') as f:
        X_chunk = pickle.load(f)
    with open(f'y_chunk_{i}.pickle', 'rb') as f:
        y_chunk = pickle.load(f)
    
    X.extend(X_chunk)
    y.extend(y_chunk)

X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.float32).flatten()

X = X.reshape(-1, 256, 256, 1)

# Load and split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to DMatrix (optional but recommended for advanced tuning)
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# Define hyperparameters
params = {
    'objective': 'binary:logistic',  # For classification
    'max_depth': 6,
    'learning_rate': 0.1,
    'n_estimators': 100,
    'eval_metric': 'logloss'
}

# Train the model
model = xgb.XGBClassifier(**params)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
