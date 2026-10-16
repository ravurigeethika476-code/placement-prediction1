import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score
)

from sklearn.preprocessing import LabelEncoder

from xgboost import XGBClassifier


def run_xgboost(
        filepath,
        selected_columns,
        target_column
):
    # =========================================================
    # READ DATASET
    # =========================================================

    df = pd.read_csv(filepath)

    # =========================================================
    # CHECK FEATURE COLUMNS
    # =========================================================

    for column in selected_columns:

        if column not in df.columns:
            raise ValueError(
                f"Column '{column}' not found."
            )

    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' not found."
        )

    selected_data = df[
        selected_columns + [target_column]
        ].copy()

    # =========================================================
    # REMOVE MISSING VALUES
    # =========================================================

    selected_data = selected_data.dropna()

    if len(selected_data) == 0:
        raise ValueError(
            "No data available after removing missing values."
        )

    X = selected_data[
        selected_columns
    ]

    Y = selected_data[
        target_column
    ]

    if Y.nunique() < 2:
        raise ValueError(
            "Target column must contain at least "
            "two different classes."
        )

    X = pd.get_dummies(
        X,
        drop_first=True
    )

    label_encoder = LabelEncoder()

    Y = label_encoder.fit_transform(Y)

    X_train, X_test, Y_train, Y_test = train_test_split(

        X,
        Y,

        test_size=0.20,

        random_state=42,

        stratify=Y
    )

    model = XGBClassifier(

        n_estimators=100,

        learning_rate=0.1,

        max_depth=10,

        min_child_weight=1,

        subsample=1.0,

        colsample_bytree=1.0,

        objective="binary:logistic",

        eval_metric="logloss",

        random_state=42,

        n_jobs=-1
    )

    model.fit(

        X_train,

        Y_train
    )

    Y_pred = model.predict(

        X_test
    )

    Accuracy = accuracy_score(

        Y_test,

        Y_pred
    )

    Precision = precision_score(

        Y_test,

        Y_pred,

        average="weighted",

        zero_division=0
    )
    Recall = recall_score(

        Y_test,

        Y_pred,

        average="weighted",

        zero_division=0
    )

    F1 = f1_score(

        Y_test,

        Y_pred,

        average="weighted",

        zero_division=0
    )

    cm = confusion_matrix(

        Y_test,

        Y_pred
    )

    report = classification_report(

        Y_test,

        Y_pred,

        output_dict=True,

        zero_division=0
    )

    feature_importance = []

    for feature, importance in zip(

            X.columns,

            model.feature_importances_

    ):
        feature_importance.append({

            "feature": feature,

            "importance": round(

                float(importance),

                4
            )

        })

    feature_importance = sorted(

        feature_importance,

        key=lambda x: x["importance"],

        reverse=True
    )

    return {

        "features":
            selected_columns,

        "target":
            target_column,

        "total_rows":
            len(selected_data),

        "train_rows":
            len(X_train),

        "test_rows":
            len(X_test),

        # XGBoost Settings

        "n_estimators":
            100,

        "learning_rate":
            0.1,

        "max_depth":
            10,

        "min_child_weight":
            1,

        "subsample":
            1.0,

        "colsample_bytree":
            1.0,

        "accuracy":
            round(

                Accuracy,

                4
            ),

        "precision":
            round(

                Precision,

                4
            ),

        "recall":
            round(

                Recall,

                4
            ),

        "f1":
            round(

                F1,

                4
            ),

        # Feature Importance

        "feature_importance":
            feature_importance,

        # Confusion Matrix

        "confusion_matrix":
            cm.tolist(),

        # Classification Report

        "classification_report":
            report,

        "predictions":
            Y_pred.tolist(),

        "actual":
            Y_test.tolist()

    }


if __name__ == "__main__":
    import os
    import matplotlib.pyplot as plt
    import seaborn as sns

    print("=" * 70)
    print("XGBOOST CLASSIFIER: STUDENT PLACEMENT PREDICTION")
    print("=" * 70)

    candidates = [
        "student_placement_prediction_dataset_2026.csv",
        "uploads/student_placement_prediction_dataset_2026.csv",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "student_placement_prediction_dataset_2026.csv"),
    ]
    data_path = next((p for p in candidates if os.path.exists(p)), "student_placement_prediction_dataset_2026.csv")

    features = [
        "cgpa",
        "coding_skill_score",
        "aptitude_score",
        "communication_skill_score",
        "internships_count",
        "projects_count",
        "backlogs",
        "mock_interview_score"
    ]
    target = "placement_status"

    print(f"\n[+] Features: {features}")
    print(f"[+] Target  : {target}")

    res = run_xgboost(data_path, features, target)

    print("\n--- MODEL PERFORMANCE ---")
    print(f"Accuracy  : {res['accuracy'] * 100:.2f}%")
    print(f"Precision : {res['precision'] * 100:.2f}%")
    print(f"Recall    : {res['recall'] * 100:.2f}%")
    print(f"F1 Score  : {res['f1'] * 100:.2f}%")

    print("\n--- FEATURE IMPORTANCES ---")
    fi = res["feature_importance"]
    for item in fi:
        print(f"  {item['feature']:<26} : {item['importance']:.4f}")

    # Visualizations
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 1. Confusion Matrix
    cm_arr = np.array(res["confusion_matrix"])
    labels = ["Not Placed", "Placed"]
    sns.heatmap(cm_arr, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=axes[0])
    axes[0].set_title("XGBoost Confusion Matrix", fontsize=12, fontweight='bold')
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("Actual")

    # 2. Feature Importance Horizontal Bar
    features_sorted = [item["feature"] for item in reversed(fi)]
    importances_sorted = [item["importance"] for item in reversed(fi)]
    axes[1].barh(features_sorted, importances_sorted, color='#2980b9', edgecolor='black')
    axes[1].set_title("XGBoost Feature Importance (Gain)", fontsize=12, fontweight='bold')
    axes[1].set_xlabel("Importance Score")

    plt.tight_layout()
    plt.show()