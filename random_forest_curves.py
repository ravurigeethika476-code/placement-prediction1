import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import (
    train_test_split,
    learning_curve,
    validation_curve
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score
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

def run_random_forest_curves(
    filepath,
    selected_columns=None,
    target_column="placement_status"
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
            "backlogs"
        ]

    data = df[selected_columns + [target_column]].dropna()
    y_raw = data[target_column]

    # Robust target encoding
    if y_raw.dtype == object or isinstance(y_raw.iloc[0], str):
        if set(y_raw.unique()).issubset({"Placed", "Not Placed"}):
            y = y_raw.map({"Placed": 1, "Not Placed": 0}).fillna(0).astype(int)
        else:
            from sklearn.preprocessing import LabelEncoder
            y = pd.Series(LabelEncoder().fit_transform(y_raw), index=y_raw.index)
    else:
        y = y_raw.astype(int)

    # Subsample for swift computation if dataset is large
    if len(data) > 4000:
        data_sub, _, y_sub, _ = train_test_split(
            data, y, train_size=4000, random_state=42, stratify=y
        )
        X = data_sub[selected_columns]
        y = y_sub
    else:
        X = data[selected_columns]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    rf = RandomForestClassifier(n_estimators=60, max_depth=8, random_state=42)
    rf.fit(X_train, y_train)

    y_probs = rf.predict_proba(X_test)[:, 1]

    # 1. ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_probs)
    roc_auc = auc(fpr, tpr)

    # 2. Precision-Recall Curve
    prec, rec, _ = precision_recall_curve(y_test, y_probs)
    avg_prec = average_precision_score(y_test, y_probs)

    # 3. Validation Curve on max_depth (sample 1500 for fast responsiveness)
    val_n = min(1500, len(X))
    X_val = X.iloc[:val_n]
    y_val = y.iloc[:val_n]

    param_range = [2, 4, 6, 8, 10]
    train_scores, val_scores = validation_curve(
        RandomForestClassifier(n_estimators=30, random_state=42),
        X_val, y_val,
        param_name="max_depth",
        param_range=param_range,
        cv=3,
        scoring="accuracy",
        n_jobs=1
    )

    # 4. Learning Curve
    train_sizes, lc_train_scores, lc_val_scores = learning_curve(
        RandomForestClassifier(n_estimators=30, max_depth=6, random_state=42),
        X_val, y_val,
        cv=3,
        train_sizes=np.linspace(0.2, 1.0, 4),
        scoring="accuracy",
        n_jobs=1
    )

    return {
        "features": selected_columns,
        "target": target_column,
        "roc_auc": round(float(roc_auc), 4),
        "avg_precision": round(float(avg_prec), 4),
        "fpr": [round(float(v), 4) for v in fpr],
        "tpr": [round(float(v), 4) for v in tpr],
        "precision": [round(float(v), 4) for v in prec],
        "recall": [round(float(v), 4) for v in rec],
        "param_range": param_range,
        "train_val_scores": (
            [round(float(v), 4) for v in np.mean(train_scores, axis=1)],
            [round(float(v), 4) for v in np.mean(val_scores, axis=1)]
        ),
        "lc_sizes": [int(s) for s in train_sizes],
        "lc_scores": (
            [round(float(v), 4) for v in np.mean(lc_train_scores, axis=1)],
            [round(float(v), 4) for v in np.mean(lc_val_scores, axis=1)]
        )
    }

if __name__ == "__main__":
    print("=" * 70)
    print("RANDOM FOREST: PERFORMANCE EVALUATION CURVES")
    print("=" * 70)

    data_path = get_data_path()
    res = run_random_forest_curves(data_path)

    print(f"\n[+] ROC-AUC Score         : {res['roc_auc']:.4f}")
    print(f"[+] Average Precision (AP): {res['avg_precision']:.4f}")

    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))

    # 1. ROC Curve
    axes[0, 0].plot(res["fpr"], res["tpr"], color="#2980b9", lw=2.5,
                    label=f"Random Forest (AUC = {res['roc_auc']:.4f})")
    axes[0, 0].plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1.5, label="Random Guess")
    axes[0, 0].set_title("Receiver Operating Characteristic (ROC) Curve", fontsize=12, fontweight="bold")
    axes[0, 0].set_xlabel("False Positive Rate")
    axes[0, 0].set_ylabel("True Positive Rate")
    axes[0, 0].legend(loc="lower right")

    # 2. Precision-Recall Curve
    axes[0, 1].plot(res["recall"], res["precision"], color="#27ae60", lw=2.5,
                    label=f"PR Curve (AP = {res['avg_precision']:.4f})")
    axes[0, 1].set_title("Precision-Recall Curve", fontsize=12, fontweight="bold")
    axes[0, 1].set_xlabel("Recall")
    axes[0, 1].set_ylabel("Precision")
    axes[0, 1].legend(loc="lower left")

    # 3. Validation Curve (max_depth)
    p_range = res["param_range"]
    train_mean, val_mean = res["train_val_scores"]
    axes[1, 0].plot(p_range, train_mean, 'o-', color='#e74c3c', label="Training Score")
    axes[1, 0].plot(p_range, val_mean, 'o-', color='#3498db', label="Validation Score (CV)")
    axes[1, 0].set_title("Validation Curve: Impact of Tree Depth (max_depth)", fontsize=12, fontweight="bold")
    axes[1, 0].set_xlabel("max_depth")
    axes[1, 0].set_ylabel("Accuracy")
    axes[1, 0].legend()

    # 4. Learning Curve
    sizes = res["lc_sizes"]
    lc_tr_mean, lc_val_mean = res["lc_scores"]
    axes[1, 1].plot(sizes, lc_tr_mean, 'o-', color='#e74c3c', label="Training Accuracy")
    axes[1, 1].plot(sizes, lc_val_mean, 'o-', color='#2ecc71', label="Validation Accuracy")
    axes[1, 1].set_title("Learning Curve (Bias vs Variance Diagnostic)", fontsize=12, fontweight="bold")
    axes[1, 1].set_xlabel("Training Sample Size")
    axes[1, 1].set_ylabel("Accuracy")
    axes[1, 1].legend()

    plt.suptitle("Random Forest Diagnostic Curves & Model Validation", fontsize=15, fontweight="bold", y=0.99)
    plt.tight_layout()
    plt.show()
