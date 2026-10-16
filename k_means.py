import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

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

def train_kmeans_model(df):
    """Function compatible with web application."""
    feature_cols = ["cgpa", "aptitude_score", "coding_skill_score"]
    model_data = df[feature_cols].dropna().copy()

    model = KMeans(n_clusters=3, random_state=42, n_init=10)
    model.fit(model_data)

    model_data["Cluster"] = model.labels_
    centers = pd.DataFrame(model.cluster_centers_, columns=feature_cols)

    result = {
        "model": model,
        "clustered_data": model_data,
        "cluster_centers": centers,
        "clusters": 3
    }
    return result

def predict_cluster(model, cgpa, aptitude, coding):
    """Predict cluster index for given student metrics."""
    student = pd.DataFrame({
        "cgpa": [cgpa],
        "aptitude_score": [aptitude],
        "coding_skill_score": [coding]
    })
    cluster = model.predict(student)[0]
    return cluster

def run_kmeans_analysis():
    print("=" * 70)
    print("K-MEANS CLUSTERING: STUDENT SKILL & PERFORMANCE PROFILING")
    print("=" * 70)

    file_path = get_data_path()
    df = pd.read_csv(file_path)

    features = ["cgpa", "coding_skill_score", "aptitude_score"]
    data = df[features].dropna().copy()

    # Standardization
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)

    # 1. Elbow Method to find optimal K
    print("\n[+] Calculating Inertia and Silhouette scores for K=2 to 8...")
    k_range = range(2, 9)
    inertias = []
    silhouettes = []

    # Sample for fast silhouette calculation
    sample_size = min(3000, len(scaled_data))
    sample_indices = np.random.RandomState(42).choice(len(scaled_data), sample_size, replace=False)
    scaled_sample = scaled_data[sample_indices]

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(scaled_data)
        inertias.append(km.inertia_)
        sil = silhouette_score(scaled_sample, km.predict(scaled_sample))
        silhouettes.append(sil)
        print(f"  k = {k} | Inertia: {km.inertia_:>10.2f} | Silhouette Score: {sil:.4f}")

    # Plot Elbow & Silhouette
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(list(k_range), inertias, 'bo-', linewidth=2, markersize=8)
    axes[0].set_title("Elbow Method for Optimal K (Inertia)", fontsize=12, fontweight='bold')
    axes[0].set_xlabel("Number of Clusters (K)")
    axes[0].set_ylabel("Inertia (Within-Cluster Sum of Squares)")

    axes[1].plot(list(k_range), silhouettes, 'ro-', linewidth=2, markersize=8)
    axes[1].set_title("Silhouette Score vs Number of Clusters", fontsize=12, fontweight='bold')
    axes[1].set_xlabel("Number of Clusters (K)")
    axes[1].set_ylabel("Silhouette Score")

    plt.tight_layout()
    plt.show()

    # 2. Fit Final Model with K = 3
    final_k = 3
    km_final = KMeans(n_clusters=final_k, random_state=42, n_init=10)
    data["Cluster"] = km_final.fit_predict(data)

    # Label Clusters based on aggregate score
    cluster_means = data.groupby("Cluster").mean()
    sorted_clusters = cluster_means.mean(axis=1).sort_values().index.tolist()
    label_map = {
        sorted_clusters[0]: "Low Placement Potential",
        sorted_clusters[1]: "Moderate Placement Potential",
        sorted_clusters[2]: "High Placement Potential"
    }
    data["Cluster_Label"] = data["Cluster"].map(label_map)

    print("\n--- CLUSTER PROFILES (CENTROIDS) ---")
    for cluster_id in sorted_clusters:
        label = label_map[cluster_id]
        print(f"\nCluster {cluster_id} [{label}]:")
        for feat in features:
            print(f"  Avg {feat}: {cluster_means.loc[cluster_id, feat]:.2f}")

    # Plot 2D Cluster Visualizations
    plt.figure(figsize=(10, 6))
    colors = {
        "Low Placement Potential": "#e74c3c",
        "Moderate Placement Potential": "#f39c12",
        "High Placement Potential": "#27ae60"
    }
    for label, col in colors.items():
        subset = data[data["Cluster_Label"] == label]
        plt.scatter(subset["cgpa"], subset["coding_skill_score"], c=col, label=label, alpha=0.6, s=25)

    centers = km_final.cluster_centers_
    plt.scatter(centers[:, 0], centers[:, 2], c="black", marker="X", s=200, label="Centroids", edgecolor="white")
    plt.title("K-Means Student Clustering: CGPA vs Coding Skill Score", fontsize=13, fontweight='bold')
    plt.xlabel("CGPA")
    plt.ylabel("Coding Skill Score")
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_kmeans_analysis()
