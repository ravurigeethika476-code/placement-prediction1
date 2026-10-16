from flask import Flask, render_template, request, redirect, session, url_for
from linear_regression import run_linear_regression
from logistic_regression import run_logistic_regression
from decision_tree import run_decision_tree
from random_forest import run_random_forest
from xgboost_model import run_xgboost
from k_means import train_kmeans_model, predict_cluster
from sgboost import run_sgboost
from random_forest_curves import run_random_forest_curves
from pca import run_pca
from tsne_model import run_tsne
from umap_model import run_umap
import os
import io
import base64

import pandas as pd
import numpy as np

from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    Normalizer,
    LabelEncoder
)

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from werkzeug.utils import secure_filename


# =========================================================
# FLASK CONFIGURATION
# =========================================================

app = Flask(__name__)

app.secret_key = "placement_prediction"

# Register MLOps Placement Blueprint
from placement import placement_bp
app.register_blueprint(placement_bp, url_prefix="/placement")

UPLOAD_FOLDER = "uploads"
PROCESSED_FOLDER = "processed"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def get_dataset_path():

    filename = session.get("dataset")

    if filename is None:
        candidates = [
            "student_placement_prediction_dataset_2026.csv",
            os.path.join(UPLOAD_FOLDER, "student_placement_prediction_dataset_2026.csv"),
            os.path.join(PROCESSED_FOLDER, "student_placement_prediction_dataset_2026.csv")
        ]
        for c in candidates:
            if os.path.exists(c):
                session["dataset"] = os.path.basename(c)
                return c
        return None

    processed_path = os.path.join(
        PROCESSED_FOLDER,
        filename
    )

    original_path = os.path.join(
        UPLOAD_FOLDER,
        filename
    )

    # Use processed dataset if available
    if os.path.exists(processed_path):
        return processed_path

    if os.path.exists(original_path):
        return original_path

    if os.path.exists(filename):
        return filename

    # Default fallback
    default_csv = "student_placement_prediction_dataset_2026.csv"
    if os.path.exists(default_csv):
        return default_csv

    return None


def load_current_dataset():

    filepath = get_dataset_path()

    if filepath is None:
        return None

    if not os.path.exists(filepath):
        return None

    try:
        return pd.read_csv(filepath)

    except Exception:
        return None


def save_current_dataset(df):

    filename = session.get("dataset")

    if filename is None:
        return False

    filepath = os.path.join(
        PROCESSED_FOLDER,
        filename
    )

    df.to_csv(
        filepath,
        index=False
    )

    return True


# =========================================================
# HOME
# =========================================================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# =========================================================
# UPLOAD DATA
# =========================================================

@app.route(
    "/upload",
    methods=["GET", "POST"]
)
def upload():

    if request.method == "POST":

        file = request.files.get("file")

        if file is None or file.filename == "":
            return "NO FILE SELECTED"

        filename = secure_filename(
            file.filename
        )

        original_path = os.path.join(
            UPLOAD_FOLDER,
            filename
        )

        file.save(original_path)

        try:

            df = pd.read_csv(
                original_path
            )

        except Exception as e:

            return (
                "Error reading CSV file: "
                + str(e)
            )

        # Create a fresh processed copy
        processed_path = os.path.join(
            PROCESSED_FOLDER,
            filename
        )

        df.to_csv(
            processed_path,
            index=False
        )

        session["dataset"] = filename

        return redirect("/summary")

    return render_template(
        "upload.html"
    )


# =========================================================
# SUMMARY
# =========================================================

@app.route("/summary")
def summary():

    filename = session.get("dataset")

    if filename is None:
        return redirect("/upload")

    df = load_current_dataset()

    if df is None:
        return redirect("/upload")

    return render_template(

        "summary.html",

        filename=filename,

        rows=df.shape[0],

        columns=df.shape[1],

        preview=df.head(10).to_html(
            classes="table table-bordered",
            index=False
        ),

        cat_cols=df.select_dtypes(
            exclude=np.number
        ).columns.tolist()
    )


# =========================================================
# MISSING VALUES
# =========================================================

@app.route(
    "/missingvalues",
    methods=["GET", "POST"]
)
def missing_values():

    filename = session.get("dataset")

    if filename is None:
        return redirect("/upload")

    df = load_current_dataset()

    if df is None:
        return redirect("/upload")

    message = None
    message_type = "success"


    # =====================================================
    # HANDLE MISSING VALUES
    # =====================================================

    if request.method == "POST":

        column = request.form.get(
            "column"
        )

        method = request.form.get(
            "method"
        )


        if column not in df.columns:

            message = "Invalid column selected."

            message_type = "error"

        else:

            try:

                # -----------------------------------------
                # MEAN
                # -----------------------------------------

                if method == "mean":

                    if pd.api.types.is_numeric_dtype(
                        df[column]
                    ):

                        df[column] = df[column].fillna(
                            df[column].mean()
                        )

                        message = (
                            f"Missing values in "
                            f"{column} filled using Mean."
                        )

                    else:

                        message = (
                            "Mean can only be used "
                            "for numerical columns."
                        )

                        message_type = "error"


                # -----------------------------------------
                # MEDIAN
                # -----------------------------------------

                elif method == "median":

                    if pd.api.types.is_numeric_dtype(
                        df[column]
                    ):

                        df[column] = df[column].fillna(
                            df[column].median()
                        )

                        message = (
                            f"Missing values in "
                            f"{column} filled using Median."
                        )

                    else:

                        message = (
                            "Median can only be used "
                            "for numerical columns."
                        )

                        message_type = "error"


                # -----------------------------------------
                # MODE
                # -----------------------------------------

                elif method == "mode":

                    mode_value = df[column].mode()

                    if not mode_value.empty:

                        df[column] = df[column].fillna(
                            mode_value.iloc[0]
                        )

                        message = (
                            f"Missing values in "
                            f"{column} filled using Mode."
                        )

                    else:

                        message = (
                            f"Unable to calculate mode "
                            f"for {column}."
                        )

                        message_type = "error"


                # -----------------------------------------
                # ZERO
                # -----------------------------------------

                elif method == "zero":

                    df[column] = df[column].fillna(0)

                    message = (
                        f"Missing values in "
                        f"{column} filled with 0."
                    )


                # -----------------------------------------
                # FORWARD FILL
                # -----------------------------------------

                elif method == "ffill":

                    df[column] = df[column].ffill()

                    message = (
                        f"Forward fill applied "
                        f"to {column}."
                    )


                # -----------------------------------------
                # BACKWARD FILL
                # -----------------------------------------

                elif method == "bfill":

                    df[column] = df[column].bfill()

                    message = (
                        f"Backward fill applied "
                        f"to {column}."
                    )


                # -----------------------------------------
                # DROP ROWS
                # -----------------------------------------

                elif method == "drop":

                    before = len(df)

                    df = df.dropna(
                        subset=[column]
                    )

                    after = len(df)

                    removed = before - after

                    message = (
                        f"{removed} rows removed "
                        f"because of missing values "
                        f"in {column}."
                    )


                # -----------------------------------------
                # SAVE DATASET
                # -----------------------------------------

                if message_type != "error":

                    save_current_dataset(df)


            except Exception as e:

                message = (
                    "Error while handling "
                    "missing values: "
                    + str(e)
                )

                message_type = "error"


    # =====================================================
    # MISSING VALUES SUMMARY
    # =====================================================

    summary_data = pd.DataFrame({

        "Column": df.columns,

        "Data Type": [
            str(dtype)
            for dtype in df.dtypes
        ],

        "Missing Values": [
            int(value)
            for value in df.isnull().sum()
        ],

        "Missing Percentage": [

            round(
                (value / len(df)) * 100,
                2
            )
            if len(df) > 0
            else 0

            for value in df.isnull().sum()
        ]

    })


    summary_data["Status"] = summary_data[
        "Missing Values"
    ].apply(

        lambda x:
        "Missing"
        if x > 0
        else "No Missing"
    )


    total_missing = int(
        df.isnull().sum().sum()
    )

    no_missing = (
        total_missing == 0
    )


    return render_template(

        "missingvalues.html",

        missing_values=summary_data.to_dict(
            "records"
        ),

        columns=df.columns.tolist(),

        total_missing=total_missing,

        no_missing=no_missing,

        message=message,

        message_type=message_type
    )


# =========================================================
# ENCODING
# =========================================================

@app.route(
    "/encoding",
    methods=["GET", "POST"]
)
def encoding():

    filename = session.get("dataset")

    if filename is None:
        return redirect("/upload")

    df = load_current_dataset()

    if df is None:
        return redirect("/upload")


    # =====================================================
    # VARIABLES FOR HTML
    # =====================================================

    message = None

    message_type = "success"

    encoded_column = None

    encoding_method = None

    encoding_mapping = None

    new_columns = []

    before_preview = None

    after_preview = None

    table = None


    # =====================================================
    # APPLY ENCODING
    # =====================================================

    if request.method == "POST":

        column = request.form.get(
            "column"
        )

        method = request.form.get(
            "method"
        )


        # =================================================
        # CHECK COLUMN
        # =================================================

        if column not in df.columns:

            message = (
                "Invalid column selected."
            )

            message_type = "error"


        else:

            try:

                # =========================================
                # SAVE ORIGINAL COLUMN
                # =========================================

                original_values = (
                    df[column]
                    .fillna("Missing")
                    .astype(str)
                    .copy()
                )


                # =========================================
                # LABEL ENCODING
                # =========================================

                if method == "label":

                    # Replace missing values
                    df[column] = df[column].fillna(
                        "Unknown"
                    )

                    # Convert to string
                    df[column] = (
                        df[column]
                        .astype(str)
                    )

                    # Create encoder
                    encoder = LabelEncoder()

                    # Fit and transform
                    df[column] = encoder.fit_transform(
                        df[column]
                    )


                    # -------------------------------------
                    # CREATE EXACT MAPPING
                    # -------------------------------------

                    encoding_mapping = {}

                    for category, value in zip(

                        encoder.classes_,

                        encoder.transform(
                            encoder.classes_
                        )

                    ):

                        encoding_mapping[
                            str(category)
                        ] = int(value)


                    encoded_column = column

                    encoding_method = (
                        "Label Encoding"
                    )


                    # -------------------------------------
                    # AFTER VALUES
                    # -------------------------------------

                    after_values = (
                        df[column]
                        .unique()
                        .tolist()
                    )


                    # -------------------------------------
                    # MESSAGE
                    # -------------------------------------

                    message = (

                        f"'{column}' was successfully "
                        f"converted from categorical "
                        f"values to numerical values "
                        f"using Label Encoding."

                    )


                    # -------------------------------------
                    # BEFORE PREVIEW
                    # -------------------------------------

                    before_preview = pd.DataFrame({

                        column:
                        original_values.head(10)

                    }).to_html(

                        classes=(
                            "table "
                            "table-bordered"
                        ),

                        index=False
                    )


                    # -------------------------------------
                    # AFTER PREVIEW
                    # -------------------------------------

                    after_preview = pd.DataFrame({

                        column:
                        df[column].head(10)

                    }).to_html(

                        classes=(
                            "table "
                            "table-bordered"
                        ),

                        index=False
                    )


                # =========================================
                # ONE-HOT ENCODING
                # =========================================

                elif method == "onehot":

                    # Replace missing values
                    df[column] = df[column].fillna(
                        "Unknown"
                    )

                    # Store columns before encoding
                    old_columns = (
                        df.columns.tolist()
                    )


                    # -------------------------------------
                    # APPLY ONE HOT
                    # -------------------------------------

                    df = pd.get_dummies(

                        df,

                        columns=[column],

                        dtype=int
                    )


                    # -------------------------------------
                    # FIND NEW COLUMNS
                    # -------------------------------------

                    new_columns = [

                        col

                        for col in df.columns

                        if col not in old_columns

                    ]


                    encoded_column = column

                    encoding_method = (
                        "One-Hot Encoding"
                    )


                    # -------------------------------------
                    # MESSAGE
                    # -------------------------------------

                    message = (

                        f"'{column}' was successfully "
                        f"converted using One-Hot "
                        f"Encoding."

                    )


                    # -------------------------------------
                    # BEFORE PREVIEW
                    # -------------------------------------

                    before_preview = pd.DataFrame({

                        column:
                        original_values.head(10)

                    }).to_html(

                        classes=(
                            "table "
                            "table-bordered"
                        ),

                        index=False
                    )


                    # -------------------------------------
                    # AFTER PREVIEW
                    # -------------------------------------

                    preview_columns = (
                        new_columns
                    )


                    after_preview = df[
                        preview_columns
                    ].head(10).to_html(

                        classes=(
                            "table "
                            "table-bordered"
                        ),

                        index=False
                    )


                # =========================================
                # INVALID METHOD
                # =========================================

                else:

                    message = (
                        "Invalid encoding method."
                    )

                    message_type = "error"


                # =========================================
                # SAVE ENCODED DATASET
                # =========================================

                if message_type != "error":

                    save_current_dataset(
                        df
                    )


                    # =====================================
                    # COMPLETE DATASET PREVIEW
                    # =====================================

                    table = df.head(
                        10
                    ).to_html(

                        classes=(
                            "table "
                            "table-bordered "
                            "table-striped"
                        ),

                        index=False
                    )


            except Exception as e:

                message = (
                    "Encoding Error: "
                    + str(e)
                )

                message_type = "error"


    # =====================================================
    # FIND CATEGORICAL COLUMNS
    # =====================================================

    categorical_columns = df.select_dtypes(

        include=[
            "object",
            "category"
        ]

    ).columns.tolist()


    # =====================================================
    # CHECK ENCODING COMPLETION
    # =====================================================

    encoding_complete = (
        len(categorical_columns) == 0
    )


    # =====================================================
    # RENDER ENCODING PAGE
    # =====================================================

    return render_template(

        "encoding.html",

        columns=categorical_columns,

        message=message,

        message_type=message_type,

        encoding_complete=encoding_complete,

        table=table,

        encoded_column=encoded_column,

        encoding_method=encoding_method,

        encoding_mapping=encoding_mapping,

        new_columns=new_columns,

        before_preview=before_preview,

        after_preview=after_preview
    )


# =========================================================
# VISUALIZATION
# =========================================================

@app.route("/visualization")
@app.route("/visulazation")
@app.route("/visulization")
def eda():

    filename = session.get("dataset")

    if filename is None:
        return redirect("/upload")

    df = load_current_dataset()

    if df is None:
        return redirect("/upload")


    bar_graph = None

    hist_graph = None

    bar_graph = None
    hist_graph = None
    box_plot = None

    # =====================================================
    # 1. BAR GRAPH (Categorical distribution)
    # =====================================================
    branch_cols = [c for c in df.columns if c.startswith("branch_")]

    if "branch" in df.columns:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        df["branch"].fillna("Unknown").value_counts().plot(kind="bar", ax=ax, color="#3498db", edgecolor="black")
        ax.set_title("Branch Wise Students Count", fontweight="bold")
        ax.set_xlabel("Branch")
        ax.set_ylabel("Number of Students")
        ax.tick_params(axis="x", rotation=30)
        plt.tight_layout()
        buffer = io.BytesIO()
        plt.savefig(buffer, format="png", bbox_inches="tight")
        buffer.seek(0)
        bar_graph = base64.b64encode(buffer.read()).decode("utf-8")
        plt.close(fig)

    elif branch_cols:
        # If branch was one-hot encoded, aggregate the counts
        branch_counts = {c.replace("branch_", ""): int(df[c].sum()) for c in branch_cols}
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.bar(list(branch_counts.keys()), list(branch_counts.values()), color="#3498db", edgecolor="black")
        ax.set_title("Branch Wise Students Count (from Encoded Data)", fontweight="bold")
        ax.set_xlabel("Branch")
        ax.set_ylabel("Number of Students")
        ax.tick_params(axis="x", rotation=30)
        plt.tight_layout()
        buffer = io.BytesIO()
        plt.savefig(buffer, format="png", bbox_inches="tight")
        buffer.seek(0)
        bar_graph = base64.b64encode(buffer.read()).decode("utf-8")
        plt.close(fig)

    elif "placement_status" in df.columns:
        fig, ax = plt.subplots(figsize=(6, 4))
        df["placement_status"].value_counts().plot(kind="bar", ax=ax, color=["#2ecc71", "#e74c3c"], edgecolor="black")
        ax.set_title("Placement Status Count", fontweight="bold")
        ax.set_xlabel("Status")
        ax.set_ylabel("Count")
        plt.tight_layout()
        buffer = io.BytesIO()
        plt.savefig(buffer, format="png", bbox_inches="tight")
        buffer.seek(0)
        bar_graph = base64.b64encode(buffer.read()).decode("utf-8")
        plt.close(fig)

    elif any(df.select_dtypes(include=["object", "category"]).columns):
        cat_col = df.select_dtypes(include=["object", "category"]).columns[0]
        fig, ax = plt.subplots(figsize=(6, 4))
        df[cat_col].value_counts().plot(kind="bar", ax=ax, color="#3498db", edgecolor="black")
        ax.set_title(f"{cat_col.title()} Distribution", fontweight="bold")
        ax.set_xlabel(cat_col.title())
        ax.set_ylabel("Count")
        plt.tight_layout()
        buffer = io.BytesIO()
        plt.savefig(buffer, format="png", bbox_inches="tight")
        buffer.seek(0)
        bar_graph = base64.b64encode(buffer.read()).decode("utf-8")
        plt.close(fig)

    # =====================================================
    # 2. HISTOGRAM (Numerical distribution)
    # =====================================================
    num_target_col = None
    for cand in ["age", "cgpa", "coding_skill_score", "aptitude_score"]:
        if cand in df.columns and pd.api.types.is_numeric_dtype(df[cand]):
            num_target_col = cand
            break

    if num_target_col is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [c for c in numeric_cols if c != "student_id"]
        if numeric_cols:
            num_target_col = numeric_cols[0]

    if num_target_col:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.hist(df[num_target_col].dropna(), bins=10, color="#2980b9", edgecolor="black")
        ax.set_title(f"{num_target_col.replace('_', ' ').title()} Distribution", fontweight="bold")
        ax.set_xlabel(num_target_col.replace('_', ' ').title())
        ax.set_ylabel("Frequency")
        plt.tight_layout()
        buffer = io.BytesIO()
        plt.savefig(buffer, format="png", bbox_inches="tight")
        buffer.seek(0)
        hist_graph = base64.b64encode(buffer.read()).decode("utf-8")
        plt.close(fig)

    # =====================================================
    # 3. BOX PLOT (Outlier detection)
    # =====================================================
    box_target_col = None
    for cand in ["age", "cgpa", "study_hours_per_day", "coding_skill_score"]:
        if cand in df.columns and pd.api.types.is_numeric_dtype(df[cand]):
            box_target_col = cand
            break

    if box_target_col is None and num_target_col:
        box_target_col = num_target_col

    if box_target_col:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.boxplot(df[box_target_col].dropna(), patch_artist=True,
                   boxprops=dict(facecolor="#1abc9c", color="black"),
                   medianprops=dict(color="red", linewidth=2))
        ax.set_title(f"Box Plot of {box_target_col.replace('_', ' ').title()}", fontweight="bold")
        ax.set_ylabel(box_target_col.replace('_', ' ').title())
        plt.tight_layout()
        buffer = io.BytesIO()
        plt.savefig(buffer, format="png", bbox_inches="tight")
        buffer.seek(0)
        box_plot = base64.b64encode(buffer.read()).decode("utf-8")
        plt.close(fig)

    # =====================================================
    # 4. BIVARIATE ANALYSIS (Skills vs Placement)
    # =====================================================
    bivariate_plot = None
    if "placement_status" in df.columns and "cgpa" in df.columns and "coding_skill_score" in df.columns:
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
        import seaborn as sns
        palette = {"Placed": "#2ecc71", "Not Placed": "#e74c3c"}
        sample_df = df.sample(min(2000, len(df)), random_state=42)

        # Violin plot
        sns.violinplot(x="placement_status", y="cgpa", data=sample_df, ax=axes[0], palette=palette, inner="quartile")
        axes[0].set_title("CGPA vs Placement Status", fontweight="bold")

        # Scatter plot
        sns.scatterplot(x="coding_skill_score", y="cgpa", hue="placement_status", data=sample_df, ax=axes[1], palette=palette, alpha=0.6, s=30)
        axes[1].set_title("Coding Skill vs CGPA by Placement", fontweight="bold")

        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)
        bivariate_plot = base64.b64encode(buf.read()).decode("utf-8")
        plt.close(fig)

    # =====================================================
    # 5. CORRELATION HEATMAP
    # =====================================================
    heatmap_plot = None
    num_df = df.select_dtypes(include=[np.number]).copy()
    if "student_id" in num_df.columns:
        num_df.drop(columns=["student_id"], inplace=True)
    if "placement_status" in df.columns and "placement_status" not in num_df.columns:
        num_df["placement_numeric"] = df["placement_status"].map({"Placed": 1, "Not Placed": 0}).fillna(0)

    if len(num_df.columns) >= 2:
        # Pick top most relevant numeric columns for clean heatmap
        top_cols = [c for c in ["cgpa", "coding_skill_score", "aptitude_score", "communication_skill_score", "internships_count", "projects_count", "backlogs", "salary_package_lpa", "placement_numeric"] if c in num_df.columns]
        if not top_cols:
            top_cols = num_df.columns[:8].tolist()
        fig, ax = plt.subplots(figsize=(10, 6.5))
        import seaborn as sns
        corr = num_df[top_cols].corr()
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, ax=ax, linewidths=0.5)
        ax.set_title("Feature Correlation Heatmap", fontweight="bold")
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)
        heatmap_plot = base64.b64encode(buf.read()).decode("utf-8")
        plt.close(fig)

    return render_template(
        "visualization.html",
        bar_graph=bar_graph,
        hist_graph=hist_graph,
        box_plot=box_plot,
        bivariate_plot=bivariate_plot,
        heatmap_plot=heatmap_plot
    )


# =========================================================
# SCALING AND NORMALIZATION
# =========================================================

@app.route(
    "/scaling",
    methods=["GET", "POST"]
)
def scaling():

    filename = session.get("dataset")

    if filename is None:
        return redirect("/upload")

    df = load_current_dataset()

    if df is None:
        return redirect("/upload")


    table = None

    message = None

    message_type = "success"


    # =====================================================
    # APPLY SCALING
    # =====================================================

    if request.method == "POST":

        method = request.form.get(
            "method"
        )


        # ---------------------------------------------
        # GET NUMERICAL COLUMNS
        # ---------------------------------------------

        numeric_columns = df.select_dtypes(

            include=np.number

        ).columns.tolist()


        if len(numeric_columns) == 0:

            message = (
                "No numerical columns found "
                "for scaling."
            )

            message_type = "error"


        else:

            try:

                numeric_data = df[
                    numeric_columns
                ].copy()


                # -----------------------------------------
                # HANDLE MISSING VALUES
                # -----------------------------------------

                for column in numeric_data.columns:

                    if numeric_data[
                        column
                    ].isnull().any():

                        median_value = (
                            numeric_data[
                                column
                            ].median()
                        )

                        if pd.isna(
                            median_value
                        ):

                            median_value = 0

                        numeric_data[
                            column
                        ] = numeric_data[
                            column
                        ].fillna(
                            median_value
                        )


                # -----------------------------------------
                # HANDLE INFINITE VALUES
                # -----------------------------------------

                numeric_data = (
                    numeric_data.replace(
                        [np.inf, -np.inf],
                        np.nan
                    )
                )

                numeric_data = (
                    numeric_data.fillna(0)
                )


                # -----------------------------------------
                # STANDARD SCALER
                # -----------------------------------------

                if method == "standard":

                    scaler = StandardScaler()

                    scaled_data = (
                        scaler.fit_transform(
                            numeric_data
                        )
                    )

                    message = (
                        "Standard Scaling applied "
                        "successfully."
                    )


                # -----------------------------------------
                # MIN MAX SCALER
                # -----------------------------------------

                elif method == "minmax":

                    scaler = MinMaxScaler()

                    scaled_data = (
                        scaler.fit_transform(
                            numeric_data
                        )
                    )

                    message = (
                        "Min-Max Scaling applied "
                        "successfully."
                    )


                # -----------------------------------------
                # NORMALIZER
                # -----------------------------------------

                elif method == "normalize":

                    scaler = Normalizer(
                        norm="l2"
                    )

                    scaled_data = (
                        scaler.fit_transform(
                            numeric_data
                        )
                    )

                    message = (
                        "Normalization applied "
                        "successfully."
                    )


                else:

                    message = (
                        "Please select a valid "
                        "scaling method."
                    )

                    message_type = "error"

                    scaled_data = None


                # -----------------------------------------
                # SAVE RESULT
                # -----------------------------------------

                if scaled_data is not None:

                    output = pd.DataFrame(

                        scaled_data,

                        columns=numeric_columns,

                        index=df.index
                    )


                    df[numeric_columns] = output


                    save_current_dataset(
                        df
                    )


                    table = df.head(
                        10
                    ).to_html(

                        classes=(
                            "table "
                            "table-bordered "
                            "table-striped"
                        ),

                        index=False
                    )


            except Exception as e:

                message = (
                    "Scaling Error: "
                    + str(e)
                )

                message_type = "error"


    return render_template(

        "scaling.html",

        tables=table,

        message=message,

        message_type=message_type
    )


@app.route("/linear-regression", methods=["GET", "POST"])
def linear_regression():

    filename = session.get("dataset")

    if filename is None:
        return redirect("/upload")

    filepath = os.path.join(
        UPLOAD_FOLDER,
        filename
    )

    df = pd.read_csv(filepath)

    columns = df.columns.tolist()

    results = None
    error = None

    if request.method == "POST":

        selected_columns = request.form.getlist(
            "features"
        )

        target_column = request.form.get(
            "target"
        )

        if not selected_columns:
            error = "Please select at least one feature column."

        elif not target_column:
            error = "Please select a target column."

        elif target_column in selected_columns:
            error = "Target column cannot also be a feature."

        else:

            try:

                results = run_linear_regression(
                    filepath,
                    selected_columns,
                    target_column
                )

            except Exception as e:

                error = str(e)

    return render_template(
        "linear_regression.html",
        filename=filename,
        columns=columns,
        results=results,
        error=error
    )


# =========================================================
# LOGISTIC REGRESSION
# =========================================================

@app.route("/logistic-regression", methods=["GET", "POST"])
def logistic_regression():

    filename = session.get("dataset")

    # No dataset uploaded
    if filename is None:
        return redirect("/upload")

    filepath = os.path.join(
        UPLOAD_FOLDER,
        filename
    )

    # Load dataset
    df = pd.read_csv(filepath)

    # Get all columns
    columns = df.columns.tolist()

    results = None
    error = None

    # =====================================================
    # WHEN TRAIN BUTTON IS CLICKED
    # =====================================================

    if request.method == "POST":

        # Get selected X columns
        selected_columns = request.form.getlist(
            "features"
        )

        # Get Y column
        target_column = request.form.get(
            "target"
        )

        # Check features
        if not selected_columns:

            error = (
                "Please select at least one "
                "feature column."
            )

        # Check target
        elif not target_column:

            error = (
                "Please select a target column."
            )

        # X and Y should be different
        elif target_column in selected_columns:

            error = (
                "Target column cannot also "
                "be a feature column."
            )

        else:

            try:

                results = run_logistic_regression(

                    filepath,

                    selected_columns,

                    target_column

                )

            except Exception as e:

                error = str(e)

    # =====================================================
    # DISPLAY PAGE
    # =====================================================

    return render_template(

        "logistic_regression.html",

        filename=filename,

        columns=columns,

        results=results,

        error=error

    )

@app.route("/decision-tree", methods=["GET", "POST"])
def decision_tree():
    filename = session.get("dataset")

    if filename is None:
        return redirect("/upload")

    filepath = os.path.join(UPLOAD_FOLDER, filename)

    df = pd.read_csv(filepath)
    columns = df.columns.tolist()

    results = None
    error = None

    if request.method == "POST":

        selected_columns = request.form.getlist("features")
        target_column = request.form.get("target")

        if not selected_columns:
            error = "Please select at least one feature column."

        elif not target_column:
            error = "Please select a target column."

        elif target_column in selected_columns:
            error = "Target column cannot also be a feature column."

        else:
            try:
                results = run_decision_tree(
                    filepath,
                    selected_columns,
                    target_column
                )

            except Exception as e:
                error = str(e)

    return render_template(
        "decision_tree.html",
        filename=filename,
        columns=columns,
        results=results,
        error=error
    )


@app.route("/random-forest", methods=["GET", "POST"])
def random_forest():
    filename = session.get("dataset")

    if filename is None:
        return redirect("/upload")

    filepath = os.path.join(UPLOAD_FOLDER, filename)

    df = pd.read_csv(filepath)
    columns = df.columns.tolist()

    results = None
    error = None

    if request.method == "POST":

        selected_columns = request.form.getlist("features")
        target_column = request.form.get("target")

        if not selected_columns:
            error = "Please select at least one feature column."

        elif not target_column:
            error = "Please select a target column."

        elif target_column in selected_columns:
            error = "Target column cannot also be a feature column."

        else:
            try:
                results = run_random_forest(
                    filepath,
                    selected_columns,
                    target_column
                )

            except Exception as e:
                error = str(e)

    return render_template(
        "random_forest.html",
        filename=filename,
        columns=columns,
        results=results,
        error=error
    )

@app.route("/xgboost", methods=["GET", "POST"])
def xgboost():

    filename = session.get("dataset")

    if filename is None:
        return redirect("/upload")

    filepath = os.path.join(
        UPLOAD_FOLDER,
        filename
    )

    if not os.path.exists(filepath):
        return redirect("/upload")

    try:

        df = pd.read_csv(filepath)

    except Exception as e:

        return f"Error reading dataset: {str(e)}"

    columns = df.columns.tolist()

    results = None
    error = None


    if request.method == "POST":

        # Get multiple X features
        selected_columns = request.form.getlist(
            "features"
        )

        # Get Y target
        target_column = request.form.get(
            "target"
        )


        # Check X features

        if not selected_columns:

            error = (
                "Please select at least one "
                "feature column."
            )


        # Check Y target

        elif not target_column:

            error = (
                "Please select a target column."
            )


        # X and Y cannot be same

        elif target_column in selected_columns:

            error = (
                "Target column cannot also "
                "be selected as a feature."
            )


        else:

            try:

                results = run_xgboost(

                    filepath,

                    selected_columns,

                    target_column

                )

            except Exception as e:

                error = str(e)


    return render_template(

        "xgboost.html",

        filename=filename,

        columns=columns,

        results=results,

        error=error

    )


# =========================================================
# K-MEANS CLUSTERING
# =========================================================

@app.route("/k_means", methods=["GET", "POST"], endpoint="k_means_page")
@app.route("/kmean", methods=["GET", "POST"])
@app.route("/k_mean", methods=["GET", "POST"])
def k_mean():
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    filename = session.get("dataset")
    if not filename:
        return redirect(url_for("upload"))

    df = load_current_dataset()
    if df is None:
        return redirect(url_for("upload"))

    selected_k = 3
    trained = False
    metrics = None
    plot_base64 = None
    prediction_cluster = None
    prediction_profile = None
    prediction_dist = None
    predict_cgpa_val = None
    predict_aptitude_val = None
    predict_coding_val = None
    error = None

    features = ["cgpa", "aptitude_score", "coding_skill_score"]
    available_cols = [c for c in features if c in df.columns]

    if len(available_cols) < 3:
        error = "Dataset must contain 'cgpa', 'aptitude_score', and 'coding_skill_score'."
        return render_template("kmean.html", error=error)

    data = df[features].dropna().copy()
    n_samples = len(data)

    if request.method == "POST":
        predict_action = request.form.get("predict_action")

        if predict_action:
            try:
                predict_cgpa_val = float(request.form.get("predict_cgpa", 0))
                predict_aptitude_val = float(request.form.get("predict_aptitude", 0))
                predict_coding_val = float(request.form.get("predict_coding", 0))

                k_val = int(session.get("kmeans_k", 3))
                selected_k = k_val
                km = KMeans(n_clusters=k_val, random_state=42, n_init=10)
                data["Cluster"] = km.fit_predict(data[features])

                input_point = np.array([[predict_cgpa_val, predict_aptitude_val, predict_coding_val]])
                pred_c = int(km.predict(input_point)[0])
                prediction_cluster = pred_c

                centers = km.cluster_centers_
                dist = np.linalg.norm(input_point[0] - centers[pred_c])
                prediction_dist = round(float(dist), 4)

                center_ranks = np.argsort(centers.mean(axis=1))
                rank = list(center_ranks).index(pred_c)
                profile_labels = ["Low Placement Potential", "Moderate Placement Potential", "High Placement Potential"]
                if k_val == 3:
                    prediction_profile = profile_labels[rank]
                else:
                    prediction_profile = f"Tier {rank + 1} Profile"

                trained = True
            except Exception as e:
                error = f"Prediction failed: {str(e)}"
        else:
            try:
                selected_k = int(request.form.get("n_clusters", 3))
                session["kmeans_k"] = selected_k
                trained = True
            except Exception as e:
                error = f"Training failed: {str(e)}"

        if trained:
            k_val = selected_k
            km = KMeans(n_clusters=k_val, random_state=42, n_init=10)
            data["Cluster"] = km.fit_predict(data[features])
            inertia = km.inertia_

            sample_sz = min(1500, len(data))
            sub = data.sample(sample_sz, random_state=42)
            sil = float(silhouette_score(sub[features], km.predict(sub[features])))

            centers = km.cluster_centers_
            center_ranks = np.argsort(centers.mean(axis=1))
            colors = ["#e74c3c", "#f39c12", "#27ae60", "#3498db", "#9b59b6", "#1abc9c"]
            profile_labels = ["Low Placement Potential", "Moderate Placement Potential", "High Placement Potential", "Exceptional Potential", "Developing Potential", "Basic Potential"]

            cluster_summary = []
            for cid in range(k_val):
                c_data = data[data["Cluster"] == cid]
                c_cnt = len(c_data)
                rank = list(center_ranks).index(cid)
                profile_name = profile_labels[rank] if rank < len(profile_labels) else f"Cluster {cid}"
                cluster_summary.append({
                    "cluster_id": cid,
                    "profile": profile_name,
                    "color": colors[rank % len(colors)],
                    "count": c_cnt,
                    "percentage": round(c_cnt / n_samples * 100, 2),
                    "mean_cgpa": round(float(c_data["cgpa"].mean()), 2),
                    "mean_aptitude": round(float(c_data["aptitude_score"].mean()), 2),
                    "mean_coding": round(float(c_data["coding_skill_score"].mean()), 2)
                })

            metrics = {
                "inertia": inertia,
                "silhouette": sil,
                "cluster_summary": cluster_summary
            }

            plt.figure(figsize=(9, 5))
            plot_sample = data.sample(min(1200, len(data)), random_state=42)
            for c_info in cluster_summary:
                cid = c_info["cluster_id"]
                c_pts = plot_sample[plot_sample["Cluster"] == cid]
                plt.scatter(c_pts["cgpa"], c_pts["coding_skill_score"],
                            color=c_info["color"], label=f"Cluster {cid} ({c_info['profile']})",
                            alpha=0.6, s=25)

            plt.scatter(centers[:, 0], centers[:, 2], color="black", marker="X", s=180,
                        edgecolor="white", label="Centroids")
            plt.title(f"K-Means Student Clusters (K={k_val})", fontsize=12, fontweight="bold")
            plt.xlabel("CGPA")
            plt.ylabel("Coding Skill Score")
            plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
            plt.tight_layout()

            img_buf = io.BytesIO()
            plt.savefig(img_buf, format="png", dpi=100)
            plt.close()
            img_buf.seek(0)
            plot_base64 = base64.b64encode(img_buf.getvalue()).decode("utf-8")

    return render_template(
        "kmean.html",
        selected_k=selected_k,
        trained=trained,
        n_samples=n_samples,
        metrics=metrics,
        plot_base64=plot_base64,
        predict_cgpa_val=predict_cgpa_val,
        predict_aptitude_val=predict_aptitude_val,
        predict_coding_val=predict_coding_val,
        prediction_cluster=prediction_cluster,
        prediction_profile=prediction_profile,
        prediction_dist=prediction_dist,
        error=error
    )


# =========================================================
# RANDOM FOREST DIAGNOSTIC CURVES
# =========================================================

@app.route("/random-forest-curves", methods=["GET", "POST"])
def random_forest_curves():
    filepath = get_dataset_path()
    if filepath is None:
        return redirect("/upload")

    filename = session.get("dataset", os.path.basename(filepath))
    df = load_current_dataset()
    if df is None:
        try:
            df = pd.read_csv(filepath)
        except Exception:
            return redirect("/upload")

    columns = df.columns.tolist()
    results = None
    plot_base64 = None
    error = None

    if request.method == "POST":
        selected_columns = request.form.getlist("features")
        target_column = request.form.get("target")

        if not selected_columns:
            error = "Please select at least one feature column."
        elif not target_column:
            error = "Please select a target column."
        elif target_column in selected_columns:
            error = "Target column cannot also be a feature column."
        else:
            try:
                results = run_random_forest_curves(
                    filepath,
                    selected_columns=selected_columns,
                    target_column=target_column
                )

                # Diagnostic Curves: 4-panel figure
                fig, axes = plt.subplots(2, 2, figsize=(14, 10))

                # 1. ROC Curve
                axes[0, 0].plot(results["fpr"], results["tpr"], color="#2980b9", lw=2.5,
                                label=f"Random Forest (AUC = {results['roc_auc']})")
                axes[0, 0].plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1.5, label="Random Guess")
                axes[0, 0].set_title("Receiver Operating Characteristic (ROC)", fontsize=11, fontweight="bold")
                axes[0, 0].set_xlabel("False Positive Rate")
                axes[0, 0].set_ylabel("True Positive Rate")
                axes[0, 0].legend(loc="lower right")

                # 2. Precision-Recall Curve
                axes[0, 1].plot(results["recall"], results["precision"], color="#27ae60", lw=2.5,
                                label=f"PR Curve (AP = {results['avg_precision']})")
                axes[0, 1].set_title("Precision-Recall Curve", fontsize=11, fontweight="bold")
                axes[0, 1].set_xlabel("Recall")
                axes[0, 1].set_ylabel("Precision")
                axes[0, 1].legend(loc="lower left")

                # 3. Validation Curve (max_depth)
                p_range = results["param_range"]
                train_mean, val_mean = results["train_val_scores"]
                axes[1, 0].plot(p_range, train_mean, 'o-', color='#e74c3c', label="Training Score")
                axes[1, 0].plot(p_range, val_mean, 'o-', color='#3498db', label="Validation Score (CV)")
                axes[1, 0].set_title("Validation Curve: Tree Depth (max_depth)", fontsize=11, fontweight="bold")
                axes[1, 0].set_xlabel("max_depth")
                axes[1, 0].set_ylabel("Accuracy")
                axes[1, 0].legend()

                # 4. Learning Curve
                sizes = results["lc_sizes"]
                lc_tr_mean, lc_val_mean = results["lc_scores"]
                axes[1, 1].plot(sizes, lc_tr_mean, 'o-', color='#e74c3c', label="Training Accuracy")
                axes[1, 1].plot(sizes, lc_val_mean, 'o-', color='#2ecc71', label="Validation Accuracy")
                axes[1, 1].set_title("Learning Curve (Bias vs Variance)", fontsize=11, fontweight="bold")
                axes[1, 1].set_xlabel("Training Sample Size")
                axes[1, 1].set_ylabel("Accuracy")
                axes[1, 1].legend()

                plt.tight_layout()
                buf = io.BytesIO()
                plt.savefig(buf, format="png", dpi=100)
                plt.close(fig)
                buf.seek(0)
                plot_base64 = base64.b64encode(buf.read()).decode("utf-8")

            except Exception as e:
                error = str(e)

    return render_template(
        "random_forest_curves.html",
        filename=filename,
        columns=columns,
        results=results,
        plot_base64=plot_base64,
        error=error
    )


# =========================================================
# STOCHASTIC GRADIENT BOOSTING (SGBOOST)
# =========================================================

@app.route("/sgboost", methods=["GET", "POST"])
def sgboost():
    filepath = get_dataset_path()
    if filepath is None:
        return redirect("/upload")

    filename = session.get("dataset", os.path.basename(filepath))
    df = load_current_dataset()
    if df is None:
        try:
            df = pd.read_csv(filepath)
        except Exception:
            return redirect("/upload")

    columns = df.columns.tolist()
    results = None
    plot_base64 = None
    error = None
    subsample = 0.8
    n_estimators = 100
    learning_rate = 0.1
    max_depth = 4

    if request.method == "POST":
        selected_columns = request.form.getlist("features")
        target_column = request.form.get("target")

        try:
            subsample = float(request.form.get("subsample", 0.8))
            n_estimators = int(request.form.get("n_estimators", 100))
            learning_rate = float(request.form.get("learning_rate", 0.1))
            max_depth = int(request.form.get("max_depth", 4))
        except ValueError:
            pass

        if not selected_columns:
            error = "Please select at least one feature column."
        elif not target_column:
            error = "Please select a target column."
        elif target_column in selected_columns:
            error = "Target column cannot also be a feature column."
        else:
            try:
                results = run_sgboost(
                    filepath,
                    selected_columns=selected_columns,
                    target_column=target_column,
                    subsample=subsample,
                    n_estimators=n_estimators,
                    learning_rate=learning_rate,
                    max_depth=max_depth
                )

                # Generate Deviance & Confusion Matrix Visualizations
                fig, axes = plt.subplots(1, 2, figsize=(14, 5))

                # Deviance curve
                axes[0].plot(np.arange(results['n_estimators']) + 1, results['train_deviance'], 'b-', label='Train Deviance')
                axes[0].plot(np.arange(results['n_estimators']) + 1, results['test_deviance'], 'r-', label='Test Deviance')
                axes[0].set_title(f"Deviance Loss (Subsample={results['subsample']})", fontsize=11, fontweight="bold")
                axes[0].set_xlabel("Boosting Trees")
                axes[0].set_ylabel("Deviance")
                axes[0].legend()

                # Confusion matrix
                import seaborn as sns
                cm_arr = np.array(results["confusion_matrix"])
                sns.heatmap(cm_arr, annot=True, fmt="d", cmap="Purples",
                            xticklabels=["Not Placed", "Placed"], yticklabels=["Not Placed", "Placed"], ax=axes[1])
                axes[1].set_title("SGBoost Confusion Matrix", fontsize=11, fontweight="bold")
                axes[1].set_xlabel("Predicted")
                axes[1].set_ylabel("Actual")

                plt.tight_layout()
                buf = io.BytesIO()
                plt.savefig(buf, format="png", dpi=100)
                plt.close(fig)
                buf.seek(0)
                plot_base64 = base64.b64encode(buf.read()).decode("utf-8")

            except Exception as e:
                error = str(e)

    return render_template(
        "sgboost.html",
        filename=filename,
        columns=columns,
        results=results,
        plot_base64=plot_base64,
        subsample=subsample,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        error=error
    )


# =========================================================
# PRINCIPAL COMPONENT ANALYSIS (PCA)
# =========================================================

@app.route("/pca", methods=["GET", "POST"])
def pca():
    filepath = get_dataset_path()
    if filepath is None:
        return redirect("/upload")

    filename = session.get("dataset", os.path.basename(filepath))
    df = load_current_dataset()
    if df is None:
        try:
            df = pd.read_csv(filepath)
        except Exception:
            return redirect("/upload")

    columns = df.columns.tolist()
    results = None
    plot_base64 = None
    error = None

    if request.method == "POST":
        selected_columns = request.form.getlist("features")
        target_column = request.form.get("target") or "placement_status"
        n_components = int(request.form.get("n_components", 2))

        if not selected_columns:
            error = "Please select at least two numeric features for PCA."
        else:
            try:
                res = run_pca(
                    filepath,
                    selected_columns=selected_columns,
                    target_column=target_column,
                    n_components=n_components
                )

                loadings_df = res["loadings"]
                loadings_pc1 = loadings_df["PC1"].to_dict() if "PC1" in loadings_df.columns else {}
                loadings_pc2 = loadings_df["PC2"].to_dict() if "PC2" in loadings_df.columns else {}

                results = {
                    "features": res["features"],
                    "target": target_column,
                    "n_components": res["n_components"],
                    "explained_variance_ratio": res["explained_variance_ratio"],
                    "total_variance": round(sum(res["explained_variance_ratio"]), 4),
                    "loadings_pc1": loadings_pc1,
                    "loadings_pc2": loadings_pc2
                }

                # Visualizations: Scree Plot & 2D Projection Scatter
                fig, axes = plt.subplots(1, 2, figsize=(14, 5))

                # Scree Plot
                all_var = res["all_explained_variance"]
                cum_var = res["cumulative_variance"]
                comp_range = range(1, len(all_var) + 1)
                axes[0].bar(comp_range, all_var, alpha=0.6, color='#3498db', edgecolor='black', label="Individual")
                axes[0].step(comp_range, cum_var, where='mid', color='#e74c3c', linewidth=2, label="Cumulative")
                axes[0].axhline(0.80, color='gray', linestyle='--', label="80% Threshold")
                axes[0].set_title("Scree Plot & Cumulative Variance", fontsize=11, fontweight="bold")
                axes[0].set_xlabel("Principal Component Index")
                axes[0].set_ylabel("Explained Variance Ratio")
                axes[0].legend()

                # 2D Projection Scatter
                pca_data = res["pca_data"]
                sample_df = pca_data.sample(min(1500, len(pca_data)), random_state=42)
                palette = {"Placed": "#2ecc71", "Not Placed": "#e74c3c"}

                import seaborn as sns
                sns.scatterplot(
                    x="PC1", y="PC2", hue="Target", data=sample_df,
                    palette=palette, alpha=0.6, ax=axes[1], s=25
                )
                axes[1].set_title(f"2D PCA Projection ({results['total_variance']*100:.1f}% Variance)",
                                  fontsize=11, fontweight="bold")
                axes[1].set_xlabel(f"PC1 ({results['explained_variance_ratio'][0]*100:.1f}%)")
                if len(results['explained_variance_ratio']) > 1:
                    axes[1].set_ylabel(f"PC2 ({results['explained_variance_ratio'][1]*100:.1f}%)")

                plt.tight_layout()
                buf = io.BytesIO()
                plt.savefig(buf, format="png", dpi=100)
                plt.close(fig)
                buf.seek(0)
                plot_base64 = base64.b64encode(buf.read()).decode("utf-8")

            except Exception as e:
                error = str(e)

    return render_template(
        "pca.html",
        filename=filename,
        columns=columns,
        results=results,
        plot_base64=plot_base64,
        error=error
    )


# =========================================================
# t-SNE EMBEDDING
# =========================================================

@app.route("/tsne", methods=["GET", "POST"])
def tsne():
    filepath = get_dataset_path()
    if filepath is None:
        return redirect("/upload")

    filename = session.get("dataset", os.path.basename(filepath))
    df = load_current_dataset()
    if df is None:
        try:
            df = pd.read_csv(filepath)
        except Exception:
            return redirect("/upload")

    columns = df.columns.tolist()
    results = None
    plot_base64 = None
    error = None
    perplexity = 30

    if request.method == "POST":
        perplexity = int(request.form.get("perplexity", 30))
        sample_size = int(request.form.get("sample_size", 1000))
        target_column = request.form.get("target") or "placement_status"

        try:
            res = run_tsne(
                filepath,
                target_column=target_column,
                sample_size=sample_size,
                perplexity=perplexity
            )

            results = {
                "sample_size": res["sample_size"],
                "perplexity": res["perplexity"],
                "kl_divergence": res["kl_divergence"],
                "features": res["features"]
            }

            # Plot 2D manifold
            fig, ax = plt.subplots(figsize=(8.5, 5.5))
            import seaborn as sns
            palette = {"Placed": "#2ecc71", "Not Placed": "#e74c3c"}
            df_tsne = res["tsne_data"]

            sns.scatterplot(
                x="t-SNE 1", y="t-SNE 2", hue="Target", data=df_tsne,
                palette=palette, alpha=0.7, s=30, ax=ax
            )
            ax.set_title(f"t-SNE 2D Manifold Embedding (Perplexity = {perplexity})", fontsize=12, fontweight="bold")
            ax.set_xlabel("t-SNE Dimension 1")
            ax.set_ylabel("t-SNE Dimension 2")

            plt.tight_layout()
            buf = io.BytesIO()
            plt.savefig(buf, format="png", dpi=100)
            plt.close(fig)
            buf.seek(0)
            plot_base64 = base64.b64encode(buf.read()).decode("utf-8")

        except Exception as e:
            error = str(e)

    return render_template(
        "tsne.html",
        filename=filename,
        columns=columns,
        results=results,
        plot_base64=plot_base64,
        perplexity=perplexity,
        error=error
    )


# =========================================================
# UMAP MANIFOLD PROJECTION
# =========================================================

@app.route("/umap", methods=["GET", "POST"])
def umap():
    filepath = get_dataset_path()
    if filepath is None:
        return redirect("/upload")

    filename = session.get("dataset", os.path.basename(filepath))
    df = load_current_dataset()
    if df is None:
        try:
            df = pd.read_csv(filepath)
        except Exception:
            return redirect("/upload")

    columns = df.columns.tolist()
    results = None
    plot_base64 = None
    error = None
    n_neighbors = 15

    if request.method == "POST":
        n_neighbors = int(request.form.get("n_neighbors", 15))
        min_dist = float(request.form.get("min_dist", 0.1))
        sample_size = int(request.form.get("sample_size", 1500))
        target_column = request.form.get("target") or "placement_status"

        try:
            res = run_umap(
                filepath,
                target_column=target_column,
                sample_size=sample_size,
                n_neighbors=n_neighbors,
                min_dist=min_dist
            )

            results = {
                "sample_size": res["sample_size"],
                "n_neighbors": res["n_neighbors"],
                "min_dist": res["min_dist"],
                "features": res["features"]
            }

            # Plot 2D projection
            fig, ax = plt.subplots(figsize=(8.5, 5.5))
            import seaborn as sns
            palette = {"Placed": "#2ecc71", "Not Placed": "#e74c3c"}
            df_umap = res["umap_data"]

            sns.scatterplot(
                x="UMAP 1", y="UMAP 2", hue="Target", data=df_umap,
                palette=palette, alpha=0.7, s=30, ax=ax
            )
            ax.set_title(f"UMAP 2D Manifold Projection (n_neighbors={n_neighbors}, min_dist={min_dist})",
                         fontsize=12, fontweight="bold")
            ax.set_xlabel("UMAP Dimension 1")
            ax.set_ylabel("UMAP Dimension 2")

            plt.tight_layout()
            buf = io.BytesIO()
            plt.savefig(buf, format="png", dpi=100)
            plt.close(fig)
            buf.seek(0)
            plot_base64 = base64.b64encode(buf.read()).decode("utf-8")

        except Exception as e:
            error = str(e)

    return render_template(
        "umap.html",
        filename=filename,
        columns=columns,
        results=results,
        plot_base64=plot_base64,
        n_neighbors=n_neighbors,
        error=error
    )


# =========================================================
# RUN APPLICATION
# =========================================================

if __name__ == "__main__":

    app.run(
        debug=True
    )