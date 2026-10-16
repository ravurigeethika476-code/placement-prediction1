import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc
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

def run_logistic_regression(
    filepath,
    selected_columns,
    target_column
):
    df = pd.read_csv(filepath)

    for column in selected_columns:
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found.")

    if target_column not in df.columns:
        raise ValueError(f"Target '{target_column}' not found.")

    selected_data = df[selected_columns + [target_column]].copy()
    selected_data = selected_data.dropna()

    X = selected_data[selected_columns]
    Y = selected_data[target_column]

    if not all(pd.api.types.is_numeric_dtype(X[col]) for col in X.columns):
        raise ValueError("Selected feature columns must be numerical.")

    if Y.nunique() < 2:
        raise ValueError("Target column must contain at least two different classes.")

    X_train, X_test, Y_train, Y_test = train_test_split(
        X,
        Y,
        test_size=0.2,
        random_state=42,
        stratify=Y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_scaled, Y_train)

    Y_pred = model.predict(X_test_scaled)

    Accuracy = accuracy_score(Y_test, Y_pred)
    Precision = precision_score(Y_test, Y_pred, average="weighted", zero_division=0)
    Recall = recall_score(Y_test, Y_pred, average="weighted", zero_division=0)
    F1 = f1_score(Y_test, Y_pred, average="weighted", zero_division=0)

    return {
        "features": selected_columns,
        "target": target_column,
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "accuracy": round(Accuracy, 4),
        "precision": round(Precision, 4),
        "recall": round(Recall, 4),
        "f1": round(F1, 4),
        "predictions": Y_pred.tolist(),
        "actual": Y_test.tolist()
    }

if __name__ == "__main__":
    print("=" * 70)
    print("LOGISTIC REGRESSION: PLACEMENT STATUS PREDICTION")
    print("=" * 70)

    data_path = get_data_path()
    df = pd.read_csv(data_path)

    features = [
        "cgpa",
        "coding_skill_score",
        "aptitude_score",
        "communication_skill_score",
        "internships_count",
        "projects_count",
        "backlogs"
    ]
    target = "placement_status"

    print(f"\n[+] Features: {features}")
    print(f"[+] Target: {target}")

    results = run_logistic_regression(data_path, features, target)

    print(f"\n--- MODEL EVALUATION METRICS ---")
    print(f"Accuracy  : {results['accuracy'] * 100:.2f}%")
    print(f"Precision : {results['precision'] * 100:.2f}%")
    print(f"Recall    : {results['recall'] * 100:.2f}%")
    print(f"F1 Score  : {results['f1'] * 100:.2f}%")

    # Classification Report
    y_test = results["actual"]
    y_pred = results["predictions"]
    print("\n--- CLASSIFICATION REPORT ---")
    print(classification_report(y_test, y_pred, zero_division=0))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    labels = sorted(list(set(y_test)))

    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Heatmap of Confusion Matrix
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=axes[0])
    axes[0].set_title("Confusion Matrix - Logistic Regression", fontsize=12, fontweight='bold')
    axes[0].set_xlabel("Predicted Class")
    axes[0].set_ylabel("Actual Class")

    # Metrics Bar Chart
    metrics_names = ["Accuracy", "Precision", "Recall", "F1 Score"]
    metrics_vals = [results['accuracy'], results['precision'], results['recall'], results['f1']]
    bars = axes[1].bar(metrics_names, metrics_vals, color=['#3498db', '#2ecc71', '#e67e22', '#9b59b6'], edgecolor='black', width=0.5)
    axes[1].set_ylim(0, 1.1)
    axes[1].set_title("Performance Metrics Summary", fontsize=12, fontweight='bold')
    axes[1].set_ylabel("Score (0 to 1)")
    for bar in bars:
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02, f"{bar.get_height():.2%}", ha='center', fontweight='bold')

    plt.tight_layout()
    plt.show()