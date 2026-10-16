import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


def run_random_forest(
    filepath,
    selected_columns,
    target_column
):

    # =====================================================
    # 1. LOAD PREPROCESSED DATA
    # =====================================================

    df = pd.read_csv(filepath)

    if df.empty:

        raise ValueError(
            "The dataset is empty."
        )


    # =====================================================
    # 2. CHECK COLUMNS
    # =====================================================

    for column in selected_columns:

        if column not in df.columns:

            raise ValueError(
                f"Column '{column}' was not found."
            )


    if target_column not in df.columns:

        raise ValueError(
            f"Target column '{target_column}' "
            f"was not found."
        )


    # =====================================================
    # 3. TARGET CANNOT BE A FEATURE
    # =====================================================

    if target_column in selected_columns:

        raise ValueError(
            "Target column cannot also be "
            "a feature column."
        )


    # =====================================================
    # 4. SELECT REQUIRED COLUMNS
    # =====================================================

    selected_data = df[
        selected_columns + [target_column]
    ].copy()


    # =====================================================
    # 5. REMOVE MISSING VALUES
    # =====================================================

    selected_data = selected_data.dropna()

    if selected_data.empty:

        raise ValueError(
            "No data remains after removing "
            "missing values."
        )


    # =====================================================
    # 6. DEFINE X AND Y
    # =====================================================

    X = selected_data[
        selected_columns
    ]

    Y = selected_data[
        target_column
    ]


    # =====================================================
    # 7. CHECK FEATURE TYPES
    # =====================================================

    for column in X.columns:

        if not pd.api.types.is_numeric_dtype(
            X[column]
        ):

            raise ValueError(
                f"Feature '{column}' is not numeric. "
                f"Please encode categorical columns "
                f"before Random Forest."
            )


    # =====================================================
    # 8. CONVERT TARGET IF CATEGORICAL
    # =====================================================

    if not pd.api.types.is_numeric_dtype(Y):

        Y = Y.astype(str)

        class_names = sorted(
            Y.unique()
        )

        class_mapping = {
            name: index
            for index, name
            in enumerate(class_names)
        }

        Y = Y.map(
            class_mapping
        )


    # =====================================================
    # 9. CHECK CLASSES
    # =====================================================

    unique_classes = Y.nunique()

    if unique_classes < 2:

        raise ValueError(
            "Random Forest requires at least "
            "two classes in the target column."
        )


    # =====================================================
    # 10. CHECK CLASS COUNTS
    # =====================================================

    class_counts = Y.value_counts()

    if class_counts.min() < 2:

        raise ValueError(
            "Some classes contain fewer than "
            "2 samples. Please select a target "
            "with enough samples in each class."
        )


    # =====================================================
    # 11. SPLIT DATASET
    # =====================================================

    X_train, X_test, Y_train, Y_test = train_test_split(

        X,

        Y,

        test_size=0.2,

        random_state=42,

        stratify=Y

    )


    # =====================================================
    # 12. CREATE RANDOM FOREST MODEL
    # =====================================================

    model = RandomForestClassifier(

        n_estimators=100,

        random_state=42,

        max_depth=10,

        n_jobs=-1

    )


    # =====================================================
    # 13. TRAIN MODEL
    # =====================================================

    model.fit(

        X_train,

        Y_train

    )


    # =====================================================
    # 14. PREDICT TEST DATA
    # =====================================================

    Y_pred = model.predict(

        X_test

    )


    # =====================================================
    # 15. EVALUATION METRICS
    # =====================================================

    accuracy = accuracy_score(

        Y_test,

        Y_pred

    )


    precision = precision_score(

        Y_test,

        Y_pred,

        average="weighted",

        zero_division=0

    )


    recall = recall_score(

        Y_test,

        Y_pred,

        average="weighted",

        zero_division=0

    )


    f1 = f1_score(

        Y_test,

        Y_pred,

        average="weighted",

        zero_division=0

    )


    # =====================================================
    # 16. CONFUSION MATRIX
    # =====================================================

    cm = confusion_matrix(

        Y_test,

        Y_pred

    )


    # =====================================================
    # 17. FEATURE IMPORTANCE
    # =====================================================

    feature_importance = {}

    for feature, importance in zip(

        selected_columns,

        model.feature_importances_

    ):

        feature_importance[feature] = round(

            float(importance),

            4

        )


    # =====================================================
    # 18. RETURN RESULTS
    # =====================================================

    return {

        "features": selected_columns,

        "target": target_column,

        "train_rows": len(X_train),

        "test_rows": len(X_test),

        "accuracy": round(
            accuracy,
            4
        ),

        "precision": round(
            precision,
            4
        ),

        "recall": round(
            recall,
            4
        ),

        "f1": round(
            f1,
            4
        ),

        "actual": Y_test.tolist(),

        "predictions": Y_pred.tolist(),

        "confusion_matrix": cm.tolist(),

        "feature_importance": feature_importance

    }


if __name__ == "__main__":
    import os
    import matplotlib.pyplot as plt
    import seaborn as sns

    print("=" * 70)
    print("RANDOM FOREST CLASSIFIER: STUDENT PLACEMENT PREDICTION")
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
        "hackathons_participated"
    ]
    target = "placement_status"

    print(f"\n[+] Features: {features}")
    print(f"[+] Target  : {target}")

    res = run_random_forest(data_path, features, target)

    print("\n--- MODEL PERFORMANCE ---")
    print(f"Accuracy  : {res['accuracy'] * 100:.2f}%")
    print(f"Precision : {res['precision'] * 100:.2f}%")
    print(f"Recall    : {res['recall'] * 100:.2f}%")
    print(f"F1 Score  : {res['f1'] * 100:.2f}%")

    print("\n--- FEATURE IMPORTANCES ---")
    fi = res["feature_importance"]
    for feat, imp in sorted(fi.items(), key=lambda x: x[1], reverse=True):
        print(f"  {feat:<26} : {imp:.4f}")

    # Visualizations
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 1. Confusion Matrix
    cm_arr = np.array(res["confusion_matrix"])
    labels = sorted(list(set(res["actual"])))
    sns.heatmap(cm_arr, annot=True, fmt="d", cmap="Greens", xticklabels=labels, yticklabels=labels, ax=axes[0])
    axes[0].set_title("Random Forest Confusion Matrix", fontsize=12, fontweight='bold')
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("Actual")

    # 2. Feature Importance Horizontal Bar
    sorted_fi = dict(sorted(fi.items(), key=lambda x: x[1]))
    axes[1].barh(list(sorted_fi.keys()), list(sorted_fi.values()), color='#27ae60', edgecolor='black')
    axes[1].set_title("Feature Importances (Gini Impurity)", fontsize=12, fontweight='bold')
    axes[1].set_xlabel("Importance Score")

    plt.tight_layout()
    plt.show()