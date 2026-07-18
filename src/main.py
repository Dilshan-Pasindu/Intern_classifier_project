import os
import sys
import joblib

sys.path.append(os.path.dirname(__file__))

from data_loader import load_dataset, inspect_dataset
from eda import run_eda
from preprocess import split_features_target, encode_categoricals, make_train_test_split, scale_features
from train import tune_knn, tune_decision_tree, tune_logistic_regression, tune_random_forest
from evaluate import evaluate_model, plot_roc_curve, plot_feature_importance

DATA_PATH = "data/my_dataset.csv"


def main():
    os.makedirs("models", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    # ---------- INPUT ----------
    df = load_dataset(DATA_PATH)
    inspect_dataset(df)
    run_eda(df)

    X, y = split_features_target(df)
    X = encode_categoricals(X)
    feature_names = X.columns.tolist()

    # ---------- PROCESS ----------
    X_train, X_test, y_train, y_test = make_train_test_split(X, y)
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

    print("\nTuning models (this can take a minute )...\n")
    models = {
        "KNN": tune_knn(X_train_scaled, y_train),
        "Decision Tree": tune_decision_tree(X_train_scaled, y_train),
        "Logistic Regression": tune_logistic_regression(X_train_scaled, y_train),
        "Random Forest": tune_random_forest(X_train_scaled, y_train),
    }

    # ---------- OUTPUT ----------
    results = {}
    for name, model in models.items():
        safe_name = name.replace(" ", "_")
        metrics = evaluate_model(
            model, X_test_scaled, y_test, model_name=name,
            save_path=f"outputs/confusion_matrix_{safe_name}.png",
        )
        plot_roc_curve(
            model, X_test_scaled, y_test, model_name=name,
            save_path=f"outputs/roc_curve_{safe_name}.png",
        )
        plot_feature_importance(
            model, feature_names, model_name=name,
            save_path=f"outputs/feature_importance_{safe_name}.png",
        )
        results[name] = metrics

    print("\n=== SUMMARY (best F1 first) ===")
    for name, m in sorted(results.items(), key=lambda x: x[1]["f1"], reverse=True):
        print(f"{name:22s} F1={m['f1']:.3f}  Acc={m['accuracy']:.3f}")

    best_name = max(results, key=lambda n: results[n]["f1"])
    best_model = models[best_name]
    joblib.dump(best_model, "models/best_model.joblib")
    joblib.dump(scaler, "models/scaler.joblib")
    print(f"\nSaved best model ('{best_name}') to models/best_model.joblib")


if __name__ == "__main__":
    main()