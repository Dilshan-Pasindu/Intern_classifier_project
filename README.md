# Heart Disease Classifier

A simple machine learning project that trains and evaluates a binary classifier for heart disease risk using tabular patient data, then serves a small Streamlit demo for interactive prediction.

## Overview

This project includes:

- dataset loading and inspection
- exploratory data analysis (EDA)
- preprocessing and feature scaling
- model tuning for several classifiers
- evaluation with metrics and plots
- a saved trained model and scaler for inference
- a small Streamlit web app for making predictions

## Project Structure

```text
.
├── data/
│   └── my_dataset.csv
├── models/
│   ├── best_model.joblib
│   └── scaler.joblib
├── outputs/
│   ├── confusion_matrix_*.png
│   ├── roc_curve_*.png
│   └── feature_importance_*.png
├── src/
│   ├── app.py              # Streamlit front-end
│   ├── data_loader.py      # Dataset loading utilities
│   ├── eda.py              # Exploratory analysis
│   ├── evaluate.py         # Evaluation metrics and plots
│   ├── main.py             # End-to-end training workflow
│   ├── predict.py          # Single-record prediction utility
│   ├── preprocess.py       # Feature split + encoding + scaling
│   └── train.py            # Model tuning definitions
└── requirements.txt
```

## Dependencies

Install the required Python packages:

```bash
pip install pandas scikit-learn streamlit matplotlib seaborn joblib
```

## Running the Training Pipeline

From the project root:

```bash
python src/main.py
```

This will:

1. load the dataset from `data/my_dataset.csv`
2. inspect and analyze the data
3. preprocess features
4. train and tune multiple classifiers
5. evaluate them and save the best model to `models/best_model.joblib`
6. save the scaler to `models/scaler.joblib`

## Running the Streamlit App

After training the model:

```bash
streamlit run src/app.py
```

The app lets you input patient attributes and predicts whether the patient is at low or high risk of heart disease.

## Using the Prediction Script

You can also run a quick single-record prediction example:

```bash
python src/predict.py
```

The script currently uses a built-in sample patient record and prints the prediction result.

## Output Artifacts

The training pipeline stores evaluation artifacts in the `outputs/` directory, including:

- confusion matrices
- ROC curves
- feature importance plots

## Notes

- The project expects the dataset to contain the heart-disease-related features used by the training and UI code.
- The app is designed for demonstration and local use.
