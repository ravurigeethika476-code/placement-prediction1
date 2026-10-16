import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

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

def run_linear_regression(
    filepath,
    selected_columns,
    target_column
):
    # 1. LOAD PREPROCESSED DATA
    df = pd.read_csv(filepath)

    # Check columns
    for column in selected_columns:
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found.")

    if target_column not in df.columns:
        raise ValueError(f"Target '{target_column}' not found.")

    # 2. SELECT REQUIRED COLUMNS
    selected_data = df[selected_columns + [target_column]].copy()

    # Remove missing rows
    selected_data = selected_data.dropna()

    # Make sure selected features are numerical
    X = selected_data[selected_columns]
    Y = selected_data[target_column]

    if not all(pd.api.types.is_numeric_dtype(X[col]) for col in X.columns):
        raise ValueError("Selected feature columns must be numerical.")

    if not pd.api.types.is_numeric_dtype(Y):
        raise ValueError("Target column must be numerical for Linear Regression.")

    # 3. SPLIT DATASET
    X_train, X_test, Y_train, Y_test = train_test_split(
        X,
        Y,
        test_size=0.2,
        random_state=42
    )

    # 4. CREATE LINEAR REGRESSION MODEL
    model = LinearRegression()

    # 5. TRAIN MODEL
    model.fit(X_train, Y_train)

    # 6. PREDICT TEST DATA
    Y_pred = model.predict(X_test)

    # 7. EVALUATION METRICS
    MAE = mean_absolute_error(Y_test, Y_pred)
    MSE = mean_squared_error(Y_test, Y_pred)
    RMSE = np.sqrt(MSE)
    R2 = r2_score(Y_test, Y_pred)

    return {
        "features": selected_columns,
        "target": target_column,
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "MAE": round(MAE, 4),
        "MSE": round(MSE, 4),
        "RMSE": round(RMSE, 4),
        "R2": round(R2, 4),
        "predictions": Y_pred.tolist(),
        "actual": Y_test.tolist()
    }

if __name__ == "__main__":
    print("=" * 70)
    print("LINEAR REGRESSION: SALARY PACKAGE PREDICTION")
    print("=" * 70)

    data_path = get_data_path()
    features = [
        "cgpa",
        "coding_skill_score",
        "aptitude_score",
        "communication_skill_score",
        "internships_count",
        "projects_count"
    ]
    target = "salary_package_lpa"

    print(f"\n[+] Features: {features}")
    print(f"[+] Target: {target}")

    results = run_linear_regression(data_path, features, target)

    print(f"\n--- MODEL EVALUATION METRICS ---")
    print(f"Mean Absolute Error (MAE)  : {results['MAE']}")
    print(f"Mean Squared Error (MSE)   : {results['MSE']}")
    print(f"Root Mean Squared Error    : {results['RMSE']}")
    print(f"R² Score                   : {results['R2']}")

    # Plot Actual vs Predicted
    actual = np.array(results["actual"])
    pred = np.array(results["predictions"])

    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 1. Actual vs Predicted
    sample_idx = np.random.RandomState(42).choice(len(actual), min(1000, len(actual)), replace=False)
    axes[0].scatter(actual[sample_idx], pred[sample_idx], alpha=0.5, color='#2980b9')
    max_val = max(actual.max(), pred.max())
    min_val = min(actual.min(), pred.min())
    axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label="Perfect Prediction")
    axes[0].set_title("Actual vs Predicted Salary Package (LPA)", fontsize=12, fontweight='bold')
    axes[0].set_xlabel("Actual Salary (LPA)")
    axes[0].set_ylabel("Predicted Salary (LPA)")
    axes[0].legend()

    # 2. Residual Distribution
    residuals = actual - pred
    axes[1].hist(residuals, bins=30, color='#16a085', edgecolor='black', alpha=0.7)
    axes[1].axvline(0, color='red', linestyle='--', linewidth=2)
    axes[1].set_title("Residuals Distribution (Actual - Predicted)", fontsize=12, fontweight='bold')
    axes[1].set_xlabel("Residual Error")
    axes[1].set_ylabel("Frequency")

    plt.tight_layout()
    plt.show()