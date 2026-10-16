import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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

def run_eda2():
    print("=" * 70)
    print("EDA 2: UNIVARIATE ANALYSIS (NUMERICAL DISTRIBUTIONS & OUTLIERS)")
    print("=" * 70)

    file_path = get_data_path()
    df = pd.read_csv(file_path)

    features = [
        "cgpa",
        "coding_skill_score",
        "aptitude_score",
        "communication_skill_score",
        "logical_reasoning_score",
        "study_hours_per_day"
    ]

    print("\n--- SKEWNESS & KURTOSIS ANALYSIS ---")
    for col in features:
        if col in df.columns:
            skew = df[col].skew()
            kurt = df[col].kurt()
            print(f"Feature: {col:<26} | Skewness: {skew:>7.3f} | Kurtosis: {kurt:>7.3f}")

    # Plot 1: Histograms with KDE for continuous variables
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    for idx, col in enumerate(features):
        if col in df.columns:
            sns.histplot(df[col], kde=True, ax=axes[idx], color='#3498db', edgecolor='black', bins=20)
            axes[idx].set_title(f"Distribution of {col.replace('_', ' ').title()}", fontsize=11, fontweight='bold')
            axes[idx].set_xlabel(col.replace('_', ' ').title())
            axes[idx].set_ylabel("Count")

    plt.suptitle("Univariate Analysis: Feature Distributions (Histograms + KDE)", fontsize=15, fontweight='bold', y=0.99)
    plt.tight_layout()
    plt.show()

    # Plot 2: Boxplots for outlier detection
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.flatten()

    for idx, col in enumerate(features):
        if col in df.columns:
            sns.boxplot(y=df[col], ax=axes[idx], color='#e67e22', width=0.4)
            axes[idx].set_title(f"Box Plot: {col.replace('_', ' ').title()}", fontsize=11, fontweight='bold')
            axes[idx].set_ylabel(col.replace('_', ' ').title())

    plt.suptitle("Univariate Analysis: Outlier Detection (Box Plots)", fontsize=15, fontweight='bold', y=0.99)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_eda2()
