import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler

try:
    import umap
except ImportError:
    import umap.umap_ as umap

def get_data_path():
    candidates = [
        "student_placement_prediction_dataset_2026.csv",
        "uploads/student_placement_prediction_dataset_2026.csv",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "student_placement_prediction_dataset_2026.csv"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads", "student_placement_prediction_dataset_2026.csv"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return "student_placement_prediction_dataset_2026.csv"

def run_umap(
    filepath,
    selected_columns=None,
    target_column="placement_status",
    sample_size=3000,
    n_neighbors=15,
    min_dist=0.1,
    random_state=42
):
    df = pd.read_csv(filepath)

    if selected_columns is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if "student_id" in numeric_cols:
            numeric_cols.remove("student_id")
        if "salary_package_lpa" in numeric_cols:
            numeric_cols.remove("salary_package_lpa")
        selected_columns = numeric_cols

    clean_df = df[selected_columns + [target_column]].dropna()

    if len(clean_df) > sample_size:
        sampled = clean_df.sample(sample_size, random_state=random_state)
    else:
        sampled = clean_df.copy()

    X = sampled[selected_columns]
    y = sampled[target_column]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=2,
        metric="euclidean",
        random_state=random_state
    )
    embedding = reducer.fit_transform(X_scaled)

    umap_df = pd.DataFrame(embedding, columns=["UMAP 1", "UMAP 2"])
    umap_df["Target"] = y.values

    return {
        "features": selected_columns,
        "sample_size": len(sampled),
        "n_neighbors": n_neighbors,
        "min_dist": min_dist,
        "umap_data": umap_df
    }

if __name__ == "__main__":
    print("=" * 70)
    print("UMAP: UNIFORM MANIFOLD APPROXIMATION AND PROJECTION")
    print("=" * 70)

    data_path = get_data_path()
    sample_n = 3000

    print(f"\n[+] Running UMAP with sample size = {sample_n}...")
    res15 = run_umap(data_path, sample_size=sample_n, n_neighbors=15, min_dist=0.1)
    res50 = run_umap(data_path, sample_size=sample_n, n_neighbors=50, min_dist=0.2)

    print(f"[+] Total Features Processed: {len(res15['features'])}")

    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    palette = {"Placed": "#2ecc71", "Not Placed": "#e74c3c"}

    # 1. n_neighbors = 15 (Local Structure Emphasis)
    df15 = res15["umap_data"]
    sns.scatterplot(
        x="UMAP 1", y="UMAP 2", hue="Target", data=df15,
        palette=palette, alpha=0.7, s=30, ax=axes[0]
    )
    axes[0].set_title("UMAP (n_neighbors=15, min_dist=0.1 - Local Focus)", fontsize=13, fontweight='bold')
    axes[0].set_xlabel("UMAP Dimension 1")
    axes[0].set_ylabel("UMAP Dimension 2")

    # 2. n_neighbors = 50 (Global Structure Emphasis)
    df50 = res50["umap_data"]
    sns.scatterplot(
        x="UMAP 1", y="UMAP 2", hue="Target", data=df50,
        palette=palette, alpha=0.7, s=30, ax=axes[1]
    )
    axes[1].set_title("UMAP (n_neighbors=50, min_dist=0.2 - Global Focus)", fontsize=13, fontweight='bold')
    axes[1].set_xlabel("UMAP Dimension 1")
    axes[1].set_ylabel("UMAP Dimension 2")

    plt.suptitle("UMAP 2D Projection: Preserving Local and Global Manifolds", fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.show()
