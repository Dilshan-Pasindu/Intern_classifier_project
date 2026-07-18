import os
import sys
import joblib

sys.path.append(os.path.dirname(__file__))

from data_loader import load_dataset, inspect_dataset
from preprocess import split_features_target, encode_categoricals, make_train_test_split, scale_features
from train import find_best_k, train_knn, train_decision_tree, train_logistic_regression
from evaluate import evaluate_model

DATA_PATH = "data/my_dataset.csv"


def main():
    os.makedirs("models", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    df = load_dataset(DATA_PATH)
    inspect_dataset(df)

    X, y = split_features_target(df)
    X = encode_categoricals(X)

    X_train, X_test, y_train, y_test = make_train_test_split(X, y)
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

    print("Finding best K for KNN...")
    best_k, cv_scores = find_best_k(X_train_scaled, y_train)
    print(f"Best K: {best_k}\n")

    models = {
        f"KNN (k={best_k})": train_knn(X_train_scaled, y_train, n_neighbors=best_k),
        "Decision Tree": train_decision_tree(X_train_scaled, y_train),
        "Logistic Regression": train_logistic_regression(X_train_scaled, y_train),
    }

    results = {}
    for name, model in models.items():
        safe_name = name.replace(" ", "_").replace("(", "").replace(")", "").replace("=", "")
        metrics = evaluate_model(
            model, X_test_scaled, y_test, model_name=name,
            save_path=f"outputs/confusion_matrix_{safe_name}.png",
        )
        results[name] = metrics

    print("\n=== SUMMARY (best F1 first) ===")
    for name, m in sorted(results.items(), key=lambda x: x[1]["f1"], reverse=True):
        print(f"{name:25s} F1={m['f1']:.3f}  Acc={m['accuracy']:.3f}")

    best_name = max(results, key=lambda n: results[n]["f1"])
    best_model = models[best_name]
    joblib.dump(best_model, "models/best_model.joblib")
    joblib.dump(scaler, "models/scaler.joblib")
    print(f"\nSaved best model ('{best_name}') to models/best_model.joblib")


if __name__ == "__main__":
    main()
