from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss
import matplotlib.pyplot as plt

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train and track loss
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Predictions
pred = model.predict(X_test)
proba = model.predict_proba(X_test)

# Evaluation - THIS IS DAY 2
acc = accuracy_score(y_test, pred)
f1 = f1_score(y_test, pred, average='weighted')
loss = log_loss(y_test, proba)

print(f"✅ Accuracy: {acc:.2f}")
print(f"✅ F1-Score: {f1:.2f}")
print(f"✅ Log Loss: {loss:.4f}  <- Lower is better!")
print("\nLoss = How wrong model is. Gradient Descent minimizes it.")
