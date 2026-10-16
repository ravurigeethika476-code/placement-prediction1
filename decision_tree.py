import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


def run_decision_tree(
    filepath,
    selected_columns,
    target_column
):

    # =====================================================
    # 1. LOAD PREPROCESSED DATA
    # =====================================================

    df = pd.read_csv(filepath)

    if df.empty:
        raise ValueError("The dataset is empty.")


    # =====================================================
    # 2. CHECK SELECTED COLUMNS
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
                f"before Decision Tree."
            )


    # =====================================================
    # 8. CONVERT TARGET IF IT IS CATEGORICAL
    # =====================================================

    if not pd.api.types.is_numeric_dtype(Y):

        Y = Y.astype(str)

        class_names = sorted(
            Y.unique()
        )

        class_mapping = {
            name: index
            for index, name in enumerate(class_names)
        }

        Y = Y.map(class_mapping)


    # =====================================================
    # 9. CHECK NUMBER OF CLASSES
    # =====================================================

    unique_classes = Y.nunique()

    if unique_classes < 2:

        raise ValueError(
            "Decision Tree requires at least "
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
    # 12. CREATE DECISION TREE MODEL
    # =====================================================

    model = DecisionTreeClassifier(

        random_state=42,

        max_depth=5

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