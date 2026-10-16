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

def run_eda4():
    print("=" * 70)
    print("EDA 4: CATEGORICAL FEATURES ANALYSIS VS PLACEMENT")
    print("=" * 70)

    file_path = get_data_path()
    df = pd.read_csv(file_path)

    categorical_cols = ["branch", "college_tier", "gender", "volunteer_experience"]

    print("\n--- PLACEMENT RATES BY CATEGORY ---")
    for col in categorical_cols:
        if col in df.columns:
            ct = pd.crosstab(df[col], df["placement_status"], normalize='index') * 100
            print(f"\n[+] Placement Rate by {col.upper()}:")
            print(ct.round(2))

    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    palette = {"Placed": "#2ecc71", "Not Placed": "#e74c3c"}

    fig, axes = plt.subplots(2, 2, figsize=(15, 11))

    # 1. Branch vs Placement Status
    sns.countplot(x="branch", hue="placement_status", data=df, ax=axes[0, 0], palette=palette)
    axes[0, 0].set_title("Placement Distribution by Engineering Branch", fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel("Branch")
    axes[0, 0].set_ylabel("Student Count")

    # 2. College Tier vs Placement Status
    sns.countplot(x="college_tier", hue="placement_status", data=df, ax=axes[0, 1], palette=palette,
                  order=sorted(df["college_tier"].dropna().unique()))
    axes[0, 1].set_title("Placement Distribution by College Tier", fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel("College Tier")
    axes[0, 1].set_ylabel("Student Count")

    # 3. Gender vs Placement Status
    sns.countplot(x="gender", hue="placement_status", data=df, ax=axes[1, 0], palette=palette)
    axes[1, 0].set_title("Placement Distribution by Gender", fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel("Gender")
    axes[1, 0].set_ylabel("Student Count")

    # 4. Placement Percentage Stacked Bar by Tier
    tier_ct = pd.crosstab(df["college_tier"], df["placement_status"], normalize='index') * 100
    tier_ct.plot(kind="bar", stacked=True, ax=axes[1, 1], color=["#e74c3c", "#2ecc71"], edgecolor="black")
    axes[1, 1].set_title("Placement Proportion (%) by College Tier", fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel("College Tier")
    axes[1, 1].set_ylabel("Percentage (%)")
    axes[1, 1].legend(title="Placement Status", loc="upper right")
    axes[1, 1].tick_params(axis='x', rotation=0)

    plt.suptitle("EDA 4: Categorical Demographics & Institutional Factors", fontsize=15, fontweight='bold', y=0.99)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_eda4()
