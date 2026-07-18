import joblib
import pandas as pd

MODEL_PATH = "models/best_model.joblib"
SCALER_PATH = "models/scaler.joblib"

FEATURE_ORDER = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal",
]


def predict_one(record: dict) -> dict:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    row = pd.DataFrame([record], columns=FEATURE_ORDER)
    row_scaled = scaler.transform(row)

    prediction = model.predict(row_scaled)[0]
    result = {"prediction": int(prediction)}

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(row_scaled)[0]
        result["probabilities"] = proba.tolist()

    return result


if __name__ == "__main__":
    sample = {
        "age": 58, "sex": 1, "cp": 0, "trestbps": 130, "chol": 260,
        "fbs": 0, "restecg": 1, "thalach": 140, "exang": 1,
        "oldpeak": 2.4, "slope": 1, "ca": 1, "thal": 3,
    }
    print(predict_one(sample)) #print the prediction result for the sample input