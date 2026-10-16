import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
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

def run_logistic_regression_kfold(
    filepath,
    selected_columns,
    target_column,
    n_splits=5
):
    df = pd.read_csv(filepath)

    for col in selected_columns:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found.")

    if target_column not in df.columns:
        raise ValueError(f"Target '{target_column}' not found.")

    data = df[selected_columns + [target_column]].dropna()

    X = data[selected_columns].values
    Y_raw = data[target_column].values

    # Encode binary target (Placed: 1, Not Placed: 0)
    unique_labels = sorted(list(set(Y_raw)))
    label_to_bin = {lbl: idx for idx, lbl in enumerate(unique_labels)}
    Y = np.array([label_to_bin[val] for val in Y_raw])

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    fold_accuracies = []
    fold_precisions = []
    fold_recalls = []
    fold_f1s = []
    fold_aucs = []

    mean_fpr = np.linspace(0, 1, 100)
    tprs = []

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, Y), 1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = Y[train_idx], Y[test_idx]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X_train_scaled, y_train)

        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)

        fold_accuracies.append(acc)
        fold_precisions.append(prec)
        fold_recalls.append(rec)
        fold_f1s.append(f1)
        fold_aucs.append(roc_auc)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)

    return {
        "n_splits": n_splits,
        "features": selected_columns,
        "target": target_column,
        "fold_accuracies": [round(a, 4) for a in fold_accuracies],
        "fold_precisions": [round(p, 4) for p in fold_precisions],
        "fold_recalls": [round(r, 4) for r in fold_recalls],
        "fold_f1s": [round(f, 4) for f in fold_f1s],
        "fold_aucs": [round(u, 4) for u in fold_aucs],
        "mean_accuracy": round(float(np.mean(fold_accuracies)), 4),
        "std_accuracy": round(float(np.std(fold_accuracies)), 4),
        "mean_precision": round(float(np.mean(fold_precisions)), 4),
        "mean_recall": round(float(np.mean(fold_recalls)), 4),
        "mean_f1": round(float(np.mean(fold_f1s)), 4),
        "mean_auc": round(float(mean_auc), 4),
        "mean_fpr": mean_fpr.tolist(),
        "mean_tpr": mean_tpr.tolist()
    }

if __name__ == "__main__":
    print("=" * 70)
    print("LOGISTIC REGRESSION: STRATIFIED K-FOLD CROSS-VALIDATION")
    print("=" * 70)

    data_path = get_data_path()
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
    k = 5

    print(f"\n[+] Features : {features}")
    print(f"[+] Target   : {target}")
    print(f"[+] K-Folds  : {k}")

    res = run_logistic_regression_kfold(data_path, features, target, n_splits=k)

    print("\n--- FOLD-BY-FOLD PERFORMANCE ---")
    for i in range(k):
        print(f"Fold {i+1}: Accuracy = {res['fold_accuracies'][i]*100:.2f}% | "
              f"Precision = {res['fold_precisions'][i]*100:.2f}% | "
              f"Recall = {res['fold_recalls'][i]*100:.2f}% | "
              f"F1 = {res['fold_f1s'][i]*100:.2f}% | "
              f"AUC = {res['fold_aucs'][i]:.4f}")

    print("\n--- OVERALL CROSS-VALIDATION RESULTS ---")
    print(f"Mean Accuracy  : {res['mean_accuracy']*100:.2f}% (+/- {res['std_accuracy']*100:.2f}%)")
    print(f"Mean Precision : {res['mean_precision']*100:.2f}%")
    print(f"Mean Recall    : {res['mean_recall']*100:.2f}%")
    print(f"Mean F1 Score  : {res['mean_f1']*100:.2f}%")
    print(f"Mean ROC-AUC   : {res['mean_auc']:.4f}")

    # Visualizations
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 1. Fold Accuracies Bar Chart
    folds = [f"Fold {i+1}" for i in range(k)]
    accs = res['fold_accuracies']
    bars = axes[0].bar(folds, accs, color='#3498db', edgecolor='black', width=0.55)
    axes[0].axhline(res['mean_accuracy'], color='#e74c3c', linestyle='--', linewidth=2,
                    label=f"Mean Accuracy ({res['mean_accuracy']*100:.2f}%)")
    axes[0].set_ylim(min(accs) - 0.05, 1.0)
    axes[0].set_title(f"Accuracy Across {k}-Folds", fontsize=12, fontweight='bold')
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    for bar in bars:
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                     f"{bar.get_height():.2%}", ha='center', fontweight='bold', fontsize=9)

    # 2. Mean ROC Curve
    mean_fpr = np.array(res["mean_fpr"])
    mean_tpr = np.array(res["mean_tpr"])
    axes[1].plot(mean_fpr, mean_tpr, color='#27ae60', lw=2.5,
                 label=f"Mean ROC (AUC = {res['mean_auc']:.4f})")
    axes[1].plot([0, 1], [0, 1], linestyle='--', color='gray', lw=1.5, label="Random Guess")
    axes[1].set_title(f"Mean ROC Curve ({k}-Fold Stratified CV)", fontsize=12, fontweight='bold')
    axes[1].set_xlabel("False Positive Rate")
    axes[1].set_ylabel("True Positive Rate")
    axes[1].legend(loc="lower right")

    plt.tight_layout()
    plt.show()
