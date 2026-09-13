from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

print("Loading data...")
X, y = load_iris(return_X_y=True)
print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")

# 80% train, 20% test - this prevents overfitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Train: {len(X_train)}, Test: {len(X_test)}")

print("\nTraining model...")
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

print("Evaluating...")
pred = model.predict(X_test)
acc = accuracy_score(y_test, pred)

print(f"\n✅ Accuracy: {acc:.2f} ({acc*100:.0f}%)")
print(f"✅ Day 1 Complete - You trained your first ML model!")
