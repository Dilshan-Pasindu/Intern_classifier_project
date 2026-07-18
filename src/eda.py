import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import sys
import os

sys.path.append(os.path.dirname(__file__))
from data_loader import load_dataset, TARGET_COL


def run_eda(df: pd.DataFrame, output_dir: str = "outputs"):
    os.makedirs(output_dir, exist_ok=True)

    # Correlation heatmap
    plt.figure(figsize=(10, 8))
    numeric_df = df.select_dtypes(include="number")
    sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0)
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/correlation_heatmap.png", dpi=150)
    plt.close()
    print(f"Saved: {output_dir}/correlation_heatmap.png")

    # Distribution of each numeric feature, split by target class
    feature_cols = [c for c in numeric_df.columns if c != TARGET_COL]
    n_cols = 4
    n_rows = -(-len(feature_cols) // n_cols)  # ceiling division
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 3.5 * n_rows))
    axes = axes.flatten()

    for ax, col in zip(axes, feature_cols):
        sns.histplot(data=df, x=col, hue=TARGET_COL, kde=True, ax=ax, legend=False)
        ax.set_title(col)

    for ax in axes[len(feature_cols):]:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(f"{output_dir}/feature_distributions.png", dpi=150)
    plt.close()
    print(f"Saved: {output_dir}/feature_distributions.png")

    # Target correlation ranking - which features relate most to the target
    if TARGET_COL in numeric_df.columns:
        target_corr = numeric_df.corr()[TARGET_COL].drop(TARGET_COL).sort_values(key=abs, ascending=False)
        print("\nFeatures most correlated with target :")
        print(target_corr)


if __name__ == "__main__":
    df = load_dataset()
    run_eda(df)