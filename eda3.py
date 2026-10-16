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

def run_eda3():
    print("=" * 70)
    print("EDA 3: BIVARIATE ANALYSIS (ACADEMICS & SKILLS VS PLACEMENT)")
    print("=" * 70)

    file_path = get_data_path()
    df = pd.read_csv(file_path)

    print("\n--- COMPARISON OF MEAN SCORES: PLACED VS NOT PLACED ---")
    numeric_cols = [
        "cgpa", "coding_skill_score", "aptitude_score",
        "communication_skill_score", "logical_reasoning_score",
        "internships_count", "projects_count", "backlogs"
    ]
    mean_comparison = df.groupby("placement_status")[numeric_cols].mean().T
    print(mean_comparison.round(2))

    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    palette = {"Placed": "#2ecc71", "Not Placed": "#e74c3c"}

    # Figure 1: Academic & Skills vs Placement Status
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))

    # 1. CGPA vs Placement Status (Violin Plot)
    sns.violinplot(x="placement_status", y="cgpa", data=df, ax=axes[0, 0], palette=palette, inner="quartile")
    axes[0, 0].set_title("CGPA vs Placement Status (Violin Plot)", fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel("Placement Status")
    axes[0, 0].set_ylabel("CGPA")

    # 2. Coding Skill vs Aptitude by Placement (Scatter Plot)
    sns.scatterplot(
        x="coding_skill_score", y="aptitude_score", hue="placement_status",
        data=df.sample(min(2000, len(df)), random_state=42), ax=axes[0, 1], palette=palette, alpha=0.7
    )
    axes[0, 1].set_title("Coding Skill vs Aptitude Score by Placement", fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel("Coding Skill Score")
    axes[0, 1].set_ylabel("Aptitude Score")

    # 3. Backlogs vs Placement Status (Grouped Countplot)
    sns.countplot(x="backlogs", hue="placement_status", data=df, ax=axes[1, 0], palette=palette)
    axes[1, 0].set_title("Backlogs Count vs Placement Status", fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel("Number of Backlogs")
    axes[1, 0].set_ylabel("Count")

    # 4. Internships vs Placement Status
    sns.countplot(x="internships_count", hue="placement_status", data=df, ax=axes[1, 1], palette=palette)
    axes[1, 1].set_title("Internships Count vs Placement Status", fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel("Number of Internships")
    axes[1, 1].set_ylabel("Count")

    plt.suptitle("Bivariate Analysis: Academic & Performance Factors vs Placement", fontsize=15, fontweight='bold', y=0.99)
    plt.tight_layout()
    plt.show()

    # Figure 2: CGPA vs Salary Package for Placed Students
    placed_df = df[df["placement_status"] == "Placed"]
    if "salary_package_lpa" in placed_df.columns and (placed_df["salary_package_lpa"] > 0).any():
        plt.figure(figsize=(9, 5))
        sns.regplot(
            x="cgpa", y="salary_package_lpa",
            data=placed_df.sample(min(1500, len(placed_df)), random_state=42),
            scatter_kws={"alpha": 0.5, "color": "#16a085"}, line_kws={"color": "#c0392b", "linewidth": 2}
        )
        plt.title("CGPA vs Salary Package (LPA) for Placed Students", fontsize=13, fontweight='bold')
        plt.xlabel("CGPA")
        plt.ylabel("Salary Package (LPA)")
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    run_eda3()
