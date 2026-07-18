import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    f1_score, precision_score, recall_score,
)


def evaluate_model(model, X_test, y_test, model_name="model",
                    save_path: str = "outputs/confusion_matrix.png"):
    predictions = model.predict(X_test)

    acc = accuracy_score(y_test, predictions)
    prec = precision_score(y_test, predictions, average="weighted", zero_division=0)
    rec = recall_score(y_test, predictions, average="weighted", zero_division=0)
    f1 = f1_score(y_test, predictions, average="weighted", zero_division=0)
    cm = confusion_matrix(y_test, predictions)

    print("=" * 60)
    print(f"EVALUATION: {model_name}")
    print("=" * 60)
    print(f"Accuracy : {acc:.3f}")
    print(f"Precision: {prec:.3f}")
    print(f"Recall   : {rec:.3f}")
    print(f"F1 Score : {f1:.3f}")
    print("\nConfusion Matrix:")
    print(cm)
    print("\n", classification_report(y_test, predictions, zero_division=0))
    print("=" * 60 + "\n")

    _plot_confusion_matrix(cm, model_name, save_path)
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}


def _plot_confusion_matrix(cm, model_name, save_path):
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix — {model_name}")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved confusion matrix to : {save_path}")