import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import joblib

from src.preprocess import load_data, preprocess
# Load test data
df = load_data("data/KDDTest+.txt")
df = preprocess(df)

X = df.drop("label", axis=1)
y = df["label"]

# Load trained model
model = joblib.load("models/best_model.pkl")

# Predictions
y_pred = model.predict(X)

# Confusion matrix
cm = confusion_matrix(y, y_pred)

# Plot
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap="Blues")

plt.title("Confusion Matrix")
plt.savefig("models/confusion_matrix.png")
plt.show()