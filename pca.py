import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
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

def run_pca(
    filepath,
    selected_columns=None,
    target_column="placement_status",
    n_components=2
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
    X = clean_df[selected_columns]
    Y = clean_df[target_column]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Full PCA for scree plot
    full_pca = PCA()
    full_pca.fit(X_scaled)
    explained_variance_ratio = full_pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance_ratio)

    # 2D/3D PCA projection
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)

    pca_df = pd.DataFrame(
        X_pca,
        columns=[f"PC{i+1}" for i in range(n_components)]
    )
    pca_df["Target"] = Y.values

    # Loadings
    loadings = pd.DataFrame(
        pca.components_.T,
        columns=[f"PC{i+1}" for i in range(n_components)],
        index=selected_columns
    )

    return {
        "features": selected_columns,
        "n_components": n_components,
        "explained_variance_ratio": [round(float(v), 4) for v in pca.explained_variance_ratio_],
        "all_explained_variance": [round(float(v), 4) for v in explained_variance_ratio],
        "cumulative_variance": [round(float(v), 4) for v in cumulative_variance],
        "loadings": loadings,
        "pca_data": pca_df
    }

if __name__ == "__main__":
    print("=" * 70)
    print("PRINCIPAL COMPONENT ANALYSIS (PCA): DIMENSIONALITY REDUCTION")
    print("=" * 70)

    data_path = get_data_path()
    res = run_pca(data_path, n_components=2)

    print(f"\n[+] Total Input Features: {len(res['features'])}")
    print(f"[+] Reduced Components   : {res['n_components']}")
    print(f"[+] Explained Variance Ratio (PC1, PC2): {res['explained_variance_ratio']}")
    print(f"[+] Total Variance Explained by First 2 PCs: {sum(res['explained_variance_ratio'])*100:.2f}%")

    print("\n--- TOP FEATURE LOADINGS ON PC1 & PC2 ---")
    loadings = res["loadings"]
    print(loadings.round(4))

    # Visualizations
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. Scree Plot / Cumulative Variance
    all_var = res["all_explained_variance"]
    cum_var = res["cumulative_variance"]
    comp_range = range(1, len(all_var) + 1)

    axes[0].bar(comp_range, all_var, alpha=0.6, color='#3498db', edgecolor='black', label="Individual")
    axes[0].step(comp_range, cum_var, where='mid', color='#e74c3c', linewidth=2, label="Cumulative")
    axes[0].axhline(0.80, color='gray', linestyle='--', label="80% Threshold")
    axes[0].set_title("Scree Plot & Cumulative Explained Variance", fontsize=12, fontweight='bold')
    axes[0].set_xlabel("Principal Component Index")
    axes[0].set_ylabel("Explained Variance Ratio")
    axes[0].legend()

    # 2. 2D PCA Projection Scatter Plot
    pca_df = res["pca_data"]
    sample_df = pca_df.sample(min(1500, len(pca_df)), random_state=42)
    palette = {"Placed": "#2ecc71", "Not Placed": "#e74c3c"}

    sns.scatterplot(
        x="PC1", y="PC2", hue="Target", data=sample_df,
        palette=palette, alpha=0.6, ax=axes[1], s=25
    )
    axes[1].set_title(f"2D PCA Projection (Explains {sum(res['explained_variance_ratio'])*100:.1f}%)",
                      fontsize=12, fontweight='bold')
    axes[1].set_xlabel(f"PC1 ({res['explained_variance_ratio'][0]*100:.1f}%)")
    axes[1].set_ylabel(f"PC2 ({res['explained_variance_ratio'][1]*100:.1f}%)")

    # 3. Feature Loadings on PC1 Bar Plot
    pc1_loadings = loadings["PC1"].sort_values()
    axes[2].barh(pc1_loadings.index, pc1_loadings.values, color='#8e44ad', edgecolor='black')
    axes[2].axvline(0, color='black', linewidth=0.8)
    axes[2].set_title("Feature Contributions (Loadings) on PC1", fontsize=12, fontweight='bold')
    axes[2].set_xlabel("Loading Weight")

    plt.tight_layout()
    plt.show()
