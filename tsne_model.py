import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

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

def run_tsne(
    filepath,
    selected_columns=None,
    target_column="placement_status",
    sample_size=1500,
    perplexity=30,
    n_iter=1000,
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

    # Subsample for t-SNE performance and clarity
    if len(clean_df) > sample_size:
        sampled = clean_df.sample(sample_size, random_state=random_state)
    else:
        sampled = clean_df.copy()

    X = sampled[selected_columns]
    y = sampled[target_column]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Apply t-SNE with version-compatible parameters
    import inspect
    tsne_kwargs = {
        "n_components": 2,
        "perplexity": perplexity,
        "random_state": random_state,
        "init": "pca",
        "learning_rate": "auto"
    }
    sig = inspect.signature(TSNE.__init__)
    if "max_iter" in sig.parameters:
        tsne_kwargs["max_iter"] = n_iter
    elif "n_iter" in sig.parameters:
        tsne_kwargs["n_iter"] = n_iter

    tsne = TSNE(**tsne_kwargs)
    X_tsne = tsne.fit_transform(X_scaled)

    tsne_df = pd.DataFrame(X_tsne, columns=["t-SNE 1", "t-SNE 2"])
    tsne_df["Target"] = y.values

    return {
        "features": selected_columns,
        "sample_size": len(sampled),
        "perplexity": perplexity,
        "tsne_data": tsne_df,
        "kl_divergence": round(float(tsne.kl_divergence_), 4) if hasattr(tsne, "kl_divergence_") else None
    }

if __name__ == "__main__":
    print("=" * 70)
    print("t-SNE: NON-LINEAR MANIFOLD LEARNING & EMBEDDING")
    print("=" * 70)

    data_path = get_data_path()
    sample_n = 2000

    print(f"\n[+] Running t-SNE with sample size = {sample_n}...")
    res30 = run_tsne(data_path, sample_size=sample_n, perplexity=30)
    res50 = run_tsne(data_path, sample_size=sample_n, perplexity=50)

    print(f"[+] Input Features Count: {len(res30['features'])}")
    if res30["kl_divergence"]:
        print(f"[+] Final KL Divergence (Perplexity 30): {res30['kl_divergence']}")

    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    palette = {"Placed": "#2ecc71", "Not Placed": "#e74c3c"}

    # 1. Perplexity = 30
    df30 = res30["tsne_data"]
    sns.scatterplot(
        x="t-SNE 1", y="t-SNE 2", hue="Target", data=df30,
        palette=palette, alpha=0.7, s=35, ax=axes[0]
    )
    axes[0].set_title(f"t-SNE 2D Projection (Perplexity = 30)", fontsize=13, fontweight='bold')
    axes[0].set_xlabel("t-SNE Dimension 1")
    axes[0].set_ylabel("t-SNE Dimension 2")

    # 2. Perplexity = 50
    df50 = res50["tsne_data"]
    sns.scatterplot(
        x="t-SNE 1", y="t-SNE 2", hue="Target", data=df50,
        palette=palette, alpha=0.7, s=35, ax=axes[1]
    )
    axes[1].set_title(f"t-SNE 2D Projection (Perplexity = 50)", fontsize=13, fontweight='bold')
    axes[1].set_xlabel("t-SNE Dimension 1")
    axes[1].set_ylabel("t-SNE Dimension 2")

    plt.suptitle("t-Distributed Stochastic Neighbor Embedding (t-SNE)", fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.show()
