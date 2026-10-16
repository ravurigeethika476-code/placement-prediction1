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

def run_eda5():
    print("=" * 70)
    print("EDA 5: MULTIVARIATE ANALYSIS & CORRELATION HEATMAP")
    print("=" * 70)

    file_path = get_data_path()
    df = pd.read_csv(file_path)

    # Encode placement status to 0/1 for correlation calculation
    df_encoded = df.copy()
    if 'placement_status' in df_encoded.columns:
        df_encoded['placement_status_numeric'] = df_encoded['placement_status'].map({
            'Placed': 1, 'Not Placed': 0
        }).fillna(0).astype(int)

    # Filter numerical columns
    numeric_df = df_encoded.select_dtypes(include=[np.number])
    if 'student_id' in numeric_df.columns:
        numeric_df = numeric_df.drop(columns=['student_id'])

    corr_matrix = numeric_df.corr()

    print("\n--- TOP CORRELATIONS WITH PLACEMENT STATUS ---")
    if 'placement_status_numeric' in corr_matrix.columns:
        placement_corr = corr_matrix['placement_status_numeric'].drop('placement_status_numeric').sort_values(ascending=False)
        print(placement_corr.round(4))

    print("\n--- TOP CORRELATIONS WITH SALARY PACKAGE ---")
    if 'salary_package_lpa' in corr_matrix.columns:
        salary_corr = corr_matrix['salary_package_lpa'].drop('salary_package_lpa').sort_values(ascending=False)
        print(salary_corr.round(4))

    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

    # Figure 1: Full Correlation Heatmap
    plt.figure(figsize=(16, 12))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)

    sns.heatmap(
        corr_matrix, mask=mask, cmap=cmap, vmax=1.0, vmin=-1.0,
        annot=True, fmt=".2f", square=True, linewidths=0.5, cbar_kws={"shrink": 0.8}
    )
    plt.title("Correlation Matrix of Student Academic & Skill Features", fontsize=15, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.show()

    # Figure 2: Pairplot of Top Key Placement Drivers
    top_cols = ["cgpa", "coding_skill_score", "aptitude_score", "mock_interview_score", "placement_status"]
    available_top_cols = [c for c in top_cols if c in df.columns]
    if len(available_top_cols) >= 3:
        palette = {"Placed": "#2ecc71", "Not Placed": "#e74c3c"}
        sample_df = df.sample(min(800, len(df)), random_state=42)
        pairplot_fig = sns.pairplot(
            sample_df[available_top_cols],
            hue="placement_status",
            palette=palette,
            diag_kind="kde",
            plot_kws={"alpha": 0.6, "s": 25}
        )
        pairplot_fig.fig.suptitle("Multivariate Pairplot: Top Placement Predictors", fontsize=14, fontweight='bold', y=1.02)
        plt.show()

if __name__ == "__main__":
    run_eda5()
