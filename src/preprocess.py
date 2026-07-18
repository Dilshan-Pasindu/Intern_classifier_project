import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

TARGET_COL = "target"


def split_features_target(df: pd.DataFrame):
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    return X, y


def encode_categoricals(X: pd.DataFrame) -> pd.DataFrame:
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    if categorical_cols:
        print(f"Encoding categorical columns: {categorical_cols}")
        X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
    return X


def make_train_test_split(X, y, test_size: float = 0.2, random_state: int = 42):
    return train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
        shuffle=True,
    )


def scale_features(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler

#. 

