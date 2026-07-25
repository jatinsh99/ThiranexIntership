import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "customers.csv"
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    features = [
        "Age",
        "AnnualIncome",
        "SpendingScore",
        "MembershipTenure",
        "PurchaseFrequency",
    ]
    X = df[features]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return pd.DataFrame(X_scaled, columns=features, index=df.index)


def run_clustering(df: pd.DataFrame) -> pd.DataFrame:
    X_scaled = prepare_features(df)
    model = KMeans(n_clusters=4, random_state=42, n_init=10)
    labels = model.fit_predict(X_scaled)
    df = df.copy()
    df["Cluster"] = labels
    return df


def create_segment_summary(df: pd.DataFrame) -> pd.DataFrame:
    numeric_features = ["Age", "AnnualIncome", "SpendingScore", "MembershipTenure", "PurchaseFrequency"]
    summary = (
        df.groupby("Cluster")[numeric_features]
        .mean()
        .round(2)
    )
    summary["CustomerCount"] = df.groupby("Cluster").size().values
    summary["TopCategory"] = (
        df.groupby(["Cluster", "PreferredCategory"])
        .size()
        .groupby(level=0)
        .idxmax()
        .apply(lambda x: x[1])
    )
    summary["TopRegion"] = (
        df.groupby(["Cluster", "Region"])
        .size()
        .groupby(level=0)
        .idxmax()
        .apply(lambda x: x[1])
    )
    return summary.reset_index()


def save_outputs(df: pd.DataFrame, summary: pd.DataFrame) -> None:
    summary_path = OUTPUT_DIR / "segment_summary.csv"
    summary.to_csv(summary_path, index=False)

    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=df,
        x="AnnualIncome",
        y="SpendingScore",
        hue="Cluster",
        style="Cluster",
        s=100,
        palette="viridis",
    )
    plt.title("Customer Segments by Income and Spending")
    plt.xlabel("Annual Income")
    plt.ylabel("Spending Score")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "cluster_plot.png", dpi=300)
    plt.close()

    profile_plot = summary[["Cluster", "Age", "AnnualIncome", "SpendingScore", "PurchaseFrequency"]].melt(
        id_vars="Cluster", var_name="Metric", value_name="Value"
    )
    plt.figure(figsize=(12, 6))
    sns.barplot(data=profile_plot, x="Metric", y="Value", hue="Cluster", palette="viridis")
    plt.title("Segment Profile Comparison")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "segment_profile.png", dpi=300)
    plt.close()


def main() -> None:
    df = load_data(DATA_PATH)
    df = run_clustering(df)
    summary = create_segment_summary(df)
    save_outputs(df, summary)

    print("Customer segmentation completed successfully.")
    print("\nSegment Summary:")
    print(summary.to_string(index=False))
    print(f"\nResults saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
