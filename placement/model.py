import os
from pathlib import Path
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

BASE = Path(__file__).resolve().parent.parent
REFERENCE_DIR = BASE / "reference_data"
REFERENCE_FILE = REFERENCE_DIR / "training_reference.csv"
ARTIFACTS_DIR = BASE / "artifacts"
ARTIFACT_FILE = ARTIFACTS_DIR / "placement_artifact.joblib"

MODEL_VERSION = "1.0.0"

NUMERIC_FEATURES = [
    "age",
    "cgpa",
    "internships_count",
    "projects_count",
    "certifications_count",
    "coding_skill_score",
    "aptitude_score",
    "communication_skill_score",
    "logical_reasoning_score",
    "hackathons_participated",
    "github_repos",
    "linkedin_connections",
    "mock_interview_score",
    "attendance_percentage",
    "backlogs",
    "extracurricular_score",
    "leadership_score",
    "sleep_hours",
    "study_hours_per_day",
]

CATEGORICAL_FEATURES = [
    "gender",
    "branch",
    "college_tier",
    "volunteer_experience",
]

TARGET_COLUMN = "placement_status"


class PlacementArtifact(dict):
    """Artifact dictionary with attribute-style access for Jinja2 templates."""
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            return None


_cached_artifact = None


def train_placement_model(dataset_path=None, sample_size=15000):
    global _cached_artifact

    if dataset_path is None:
        dataset_path = BASE / "student_placement_prediction_dataset_2026.csv"
    else:
        dataset_path = Path(dataset_path)

    if not dataset_path.exists():
        raise FileNotFoundError(f"Training dataset not found at {dataset_path}")

    # Load dataset (sampled for high-speed robust training)
    df = pd.read_csv(dataset_path)
    if sample_size and len(df) > sample_size:
        train_df = df.sample(n=sample_size, random_state=42)
    else:
        train_df = df

    all_features = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    X = train_df[all_features]
    y = train_df[TARGET_COLUMN]

    # Preprocessor: Imputation + Scaling for numeric, Imputation + OneHot for categorical
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]),
                NUMERIC_FEATURES,
            ),
            (
                "cat",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("encoder", OneHotEncoder(handle_unknown="ignore")),
                ]),
                CATEGORICAL_FEATURES,
            ),
        ]
    )

    # Full pipeline: Imputation -> Encoding/Scaling -> Random Forest -> Prediction
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)),
    ])

    pipeline.fit(X, y)
    classes = list(pipeline.named_steps["classifier"].classes_)

    # Ensure directories exist
    REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    # Save reference data for drift monitoring
    # Include all training features
    reference_df = train_df[all_features]
    reference_df.to_csv(REFERENCE_FILE, index=False)

    artifact_data = {
        "version": MODEL_VERSION,
        "target_column": TARGET_COLUMN,
        "target_classes": classes,
        "features": all_features,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "pipeline": pipeline,
    }

    joblib.dump(artifact_data, ARTIFACT_FILE)
    _cached_artifact = PlacementArtifact(artifact_data)
    return _cached_artifact


def get_artifact():
    global _cached_artifact

    if _cached_artifact is not None:
        return _cached_artifact

    if ARTIFACT_FILE.exists() and REFERENCE_FILE.exists():
        try:
            data = joblib.load(ARTIFACT_FILE)
            _cached_artifact = PlacementArtifact(data)
            return _cached_artifact
        except Exception:
            pass

    return train_placement_model()


def predict_placement(input_data):
    artifact = get_artifact()
    pipeline = artifact["pipeline"]

    row = {}
    for feat in artifact["numeric_features"]:
        val = input_data.get(feat, 0)
        try:
            row[feat] = float(val)
        except (ValueError, TypeError):
            row[feat] = 0.0

    for feat in artifact["categorical_features"]:
        row[feat] = str(input_data.get(feat, "")).strip()

    df_input = pd.DataFrame([row])
    prediction = pipeline.predict(df_input)[0]

    probabilities = pipeline.predict_proba(df_input)[0]
    classes = list(pipeline.named_steps["classifier"].classes_)

    # Get probability of the predicted class
    if prediction in classes:
        pred_idx = classes.index(prediction)
        probability = float(probabilities[pred_idx])
    else:
        probability = float(max(probabilities))

    return {
        "prediction": prediction,
        "probability": probability,
        "model_version": artifact["version"],
    }
