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

def run_eda1():
    print("=" * 70)
    print("EDA 1: DATASET OVERVIEW & SUMMARY STATISTICS")
    print("=" * 70)

    file_path = get_data_path()
    df = pd.read_csv(file_path)

    print(f"\n[+] Loaded Dataset: {file_path}")
    print(f"[+] Total Rows: {df.shape[0]}, Total Columns: {df.shape[1]}")

    print("\n--- FIRST 5 RECORDS ---")
    print(df.head())

    print("\n--- DATASET COLUMNS AND DATA TYPES ---")
    print(df.dtypes)

    print("\n--- MISSING VALUES SUMMARY ---")
    missing = df.isnull().sum()
    print(missing[missing > 0] if missing.sum() > 0 else "No missing values found!")

    print(f"\n--- DUPLICATE ROWS ---: {df.duplicated().sum()}")

    print("\n--- STATISTICAL SUMMARY (NUMERICAL) ---")
    print(df.describe().T)

    print("\n--- TARGET VARIABLE DISTRIBUTION ('placement_status') ---")
    placement_counts = df['placement_status'].value_counts()
    print(placement_counts)
    placement_pct = df['placement_status'].value_counts(normalize=True) * 100
    for status, pct in placement_pct.items():
        print(f"  {status}: {pct:.2f}%")

    # Visualizations
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 1. Placement Status Count
    colors = ['#2ecc71' if x == 'Placed' else '#e74c3c' for x in placement_counts.index]
    axes[0].bar(placement_counts.index, placement_counts.values, color=colors, edgecolor='black', width=0.5)
    axes[0].set_title("Target Distribution: Placement Status", fontsize=13, fontweight='bold')
    axes[0].set_xlabel("Status", fontsize=11)
    axes[0].set_ylabel("Number of Students", fontsize=11)
    for i, v in enumerate(placement_counts.values):
        axes[0].text(i, v + (max(placement_counts.values) * 0.01), f"{v} ({v/len(df)*100:.1f}%)", ha='center', fontweight='bold')

    # 2. Placement Status Pie Chart
    axes[1].pie(placement_counts.values, labels=placement_counts.index, autopct='%1.1f%%',
                colors=['#2ecc71', '#e74c3c'], startangle=90, explode=(0.05, 0), shadow=True)
    axes[1].set_title("Placement Ratio (%)", fontsize=13, fontweight='bold')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_eda1()
