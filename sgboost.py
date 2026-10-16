import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
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

def run_sgboost(
    filepath,
    selected_columns=None,
    target_column="placement_status",
    subsample=0.8,
    n_estimators=100,
    learning_rate=0.1,
    max_depth=4,
    max_samples=10000
):
    df = pd.read_csv(filepath)

    if selected_columns is None:
        selected_columns = [
            "cgpa",
            "coding_skill_score",
            "aptitude_score",
            "communication_skill_score",
            "internships_count",
            "projects_count",
            "backlogs",
            "mock_interview_score"
        ]

    data = df[selected_columns + [target_column]].dropna()
    y_raw = data[target_column]

    if max_samples and len(data) > max_samples:
        data, _ = train_test_split(data, train_size=max_samples, random_state=42, stratify=y_raw)
        y_raw = data[target_column]

    X = data[selected_columns]

    if y_raw.dtype == object or isinstance(y_raw.iloc[0], str):
        if set(y_raw.unique()).issubset({"Placed", "Not Placed"}):
            y = y_raw.map({"Placed": 1, "Not Placed": 0}).fillna(0).astype(int)
        else:
            from sklearn.preprocessing import LabelEncoder
            y = pd.Series(LabelEncoder().fit_transform(y_raw), index=y_raw.index)
    else:
        y = y_raw.astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Stochastic Gradient Boosting (subsample < 1.0 injects stochastic randomness per tree)
    sgb = GradientBoostingClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        subsample=subsample,
        max_depth=max_depth,
        random_state=42
    )
    sgb.fit(X_train, y_train)

    y_pred = sgb.predict(X_test)
    y_prob = sgb.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    auc_score = roc_auc_score(y_test, y_prob)

    cm = confusion_matrix(y_test, y_pred)
    feature_importance = dict(zip(selected_columns, [round(float(v), 4) for v in sgb.feature_importances_]))

    # Staged deviance for test set
    test_deviance = np.zeros((n_estimators,), dtype=np.float64)
    for i, y_pred_staged in enumerate(sgb.staged_predict_proba(X_test)):
        # Binary log-loss
        eps = 1e-15
        p = np.clip(y_pred_staged[:, 1], eps, 1 - eps)
        test_deviance[i] = -2 * np.mean(y_test * np.log(p) + (1 - y_test) * np.log(1 - p))

    train_deviance = sgb.train_score_

    return {
        "features": selected_columns,
        "target": target_column,
        "subsample": subsample,
        "n_estimators": n_estimators,
        "total_rows": len(data),
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "accuracy": round(float(acc), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1": round(float(f1), 4),
        "auc": round(float(auc_score), 4),
        "confusion_matrix": cm.tolist(),
        "feature_importance": feature_importance,
        "train_deviance": [round(float(v), 4) for v in train_deviance],
        "test_deviance": [round(float(v), 4) for v in test_deviance],
        "actual": y_test.tolist(),
        "predictions": y_pred.tolist()
    }

if __name__ == "__main__":
    print("=" * 70)
    print("STOCHASTIC GRADIENT BOOSTING (SGBOOST): PLACEMENT CLASSIFICATION")
    print("=" * 70)

    data_path = get_data_path()
    res = run_sgboost(data_path, subsample=0.8, n_estimators=100)

    print(f"\n[+] Algorithm       : Stochastic Gradient Boosting (subsample = {res['subsample']})")
    print(f"[+] Trees (Estimators): {res['n_estimators']}")
    print(f"[+] Accuracy        : {res['accuracy'] * 100:.2f}%")
    print(f"[+] Precision       : {res['precision'] * 100:.2f}%")
    print(f"[+] Recall          : {res['recall'] * 100:.2f}%")
    print(f"[+] F1 Score        : {res['f1'] * 100:.2f}%")
    print(f"[+] ROC-AUC         : {res['auc']:.4f}")

    print("\n--- FEATURE IMPORTANCES ---")
    fi = res["feature_importance"]
    for feat, imp in sorted(fi.items(), key=lambda x: x[1], reverse=True):
        print(f"  {feat:<26} : {imp:.4f}")

    # Visualizations
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. Staged Deviance (Loss vs Boosting Iterations)
    axes[0].plot(np.arange(res['n_estimators']) + 1, res['train_deviance'], 'b-', label='Train Deviance')
    axes[0].plot(np.arange(res['n_estimators']) + 1, res['test_deviance'], 'r-', label='Test Deviance')
    axes[0].set_title("Deviance (Loss) Across Boosting Iterations", fontsize=12, fontweight='bold')
    axes[0].set_xlabel("Number of Boosting Trees")
    axes[0].set_ylabel("Deviance")
    axes[0].legend(loc="upper right")

    # 2. Confusion Matrix
    cm_arr = np.array(res["confusion_matrix"])
    sns.heatmap(cm_arr, annot=True, fmt="d", cmap="Purples",
                xticklabels=["Not Placed", "Placed"], yticklabels=["Not Placed", "Placed"], ax=axes[1])
    axes[1].set_title("SGBoost Confusion Matrix", fontsize=12, fontweight='bold')
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("Actual")

    # 3. Feature Importances
    sorted_fi = dict(sorted(fi.items(), key=lambda x: x[1]))
    axes[2].barh(list(sorted_fi.keys()), list(sorted_fi.values()), color='#8e44ad', edgecolor='black')
    axes[2].set_title("Relative Feature Importance", fontsize=12, fontweight='bold')
    axes[2].set_xlabel("Importance Score")

    plt.tight_layout()
    plt.show()
