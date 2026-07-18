import pandas as pd
from pathlib import Path

TARGET_COL = "target"


def load_dataset(csv_path: str = "data/my_dataset.csv") -> pd.DataFrame:
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Could not find dataset at '{csv_path}'")
    df = pd.read_csv(path)
    return df


def inspect_dataset(df: pd.DataFrame) -> None:
    print("=" * 60)
    print("DATASET OVERVIEW")
    print("=" * 60)
    print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns\n")

    print("Column data types:")
    print(df.dtypes)

    print("\nMissing values per column:")
    missing = df.isnull().sum()
    print(missing[missing > 0] if missing.sum() > 0 else "  None found ✅")

    if TARGET_COL in df.columns:
        print(f"\nClass balance ('{TARGET_COL}' column):")
        print(df[TARGET_COL].value_counts())
    else:
        print(f"\n⚠️ Warning: '{TARGET_COL}' not found in columns: {list(df.columns)}")

    print("\nFirst 5 rows:")
    print(df.head())
    print("=" * 60 + "\n")


if __name__ == "__main__":
    data = load_dataset()
    inspect_dataset(data)
