import joblib
import pandas as pd

MODEL_PATH = "models/best_model.joblib"
SCALER_PATH = "models/scaler.joblib"
FEATURE_ORDER = ["feature_1", "feature_2", "feature_3"]


def predict_one(record: dict) -> dict:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    row = pd.DataFrame([record], columns=FEATURE_ORDER)
    row_scaled = scaler.transform(row)

    prediction = model.predict(row_scaled)[0]
    result = {"prediction": prediction}

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(row_scaled)[0]
        result["probabilities"] = proba.tolist()

    return result


if __name__ == "__main__":
    sample = {"feature_1": 0, "feature_2": 0, "feature_3": 0}
    print(predict_one(sample))
