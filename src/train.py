from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

from src.preprocess import load_data, preprocess

# Load & preprocess data
df = load_data("data/KDDTrain+.txt")
df = preprocess(df)

X = df.drop("label", axis=1)
y = df["label"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Define models
models = {
    "RandomForest": RandomForestClassifier(n_estimators=100),
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "DecisionTree": DecisionTreeClassifier()
}

best_model = None
best_score = 0

print("\n📊 Model Performance:\n")

# Train and evaluate
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"{name} Accuracy: {acc:.4f}")

    print(f"\n{name} Report:\n")
    print(classification_report(y_test, y_pred))
    print("-" * 50)

    # Save best model
    if acc > best_score:
        best_score = acc
        best_model = model

# Save best model
joblib.dump(best_model, "models/best_model.pkl")

print(f"\n✅ Best Model Saved with Accuracy: {best_score:.4f}")