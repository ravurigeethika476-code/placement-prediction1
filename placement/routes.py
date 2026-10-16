import os
from pathlib import Path
from flask import Blueprint, render_template, request, redirect, url_for, flash, session
from werkzeug.utils import secure_filename

from placement.model import get_artifact, predict_placement, train_placement_model, BASE
from placement.drift import generate_report

placement_bp = Blueprint("placement", __name__, template_folder="../templates")


@placement_bp.route("/")
def index():
    try:
        artifact = get_artifact()
        return render_template("placement/index.html", artifact=artifact, error=None)
    except Exception as e:
        return render_template("placement/index.html", artifact=None, error=str(e))


@placement_bp.route("/predict", methods=["GET", "POST"])
def predict():
    try:
        artifact = get_artifact()
    except Exception as e:
        return render_template("placement/index.html", artifact=None, error=f"Model not initialized: {e}")

    if request.method == "POST":
        try:
            result = predict_placement(request.form)
            return render_template("placement/predict_result.html", result=result, artifact=artifact, input_data=request.form)
        except Exception as e:
            return render_template("placement/predict.html", artifact=artifact, error=str(e))

    return render_template("placement/predict.html", artifact=artifact, error=None)


@placement_bp.route("/monitoring", methods=["GET", "POST"])
def monitoring():
    if request.method == "POST":
        file_path = None
        use_active = request.form.get("use_active")

        if use_active:
            dataset_name = session.get("dataset")
            candidates = [
                BASE / "uploads" / dataset_name if dataset_name else None,
                BASE / "processed" / dataset_name if dataset_name else None,
                BASE / dataset_name if dataset_name else None,
                BASE / "student_placement_prediction_dataset_2026.csv",
            ]
            for c in candidates:
                if c and Path(c).exists():
                    file_path = Path(c)
                    break

            if not file_path:
                return render_template(
                    "placement/monitoring.html",
                    report=None,
                    stats=None,
                    error="No active dataset found in session. Please upload a dataset file below."
                )
        else:
            file = request.files.get("current_data")
            if not file or file.filename == "":
                return render_template(
                    "placement/monitoring.html",
                    report=None,
                    stats=None,
                    error="Please select a dataset file (.csv, .xlsx, or .xls) or click 'Audit Active Dataset'."
                )

            filename = secure_filename(file.filename)
            upload_dir = BASE / "uploads"
            upload_dir.mkdir(parents=True, exist_ok=True)
            file_path = upload_dir / f"monitoring_{filename}"
            file.save(str(file_path))

        try:
            # Ensure model and reference data exist first
            get_artifact()
            report = generate_report(str(file_path))

            # Calculate monitoring KPI summary statistics
            high_count = int((report["drift_level"] == "high").sum()) if "drift_level" in report.columns else 0
            mod_count = int((report["drift_level"] == "moderate").sum()) if "drift_level" in report.columns else 0
            low_count = int((report["drift_level"] == "low").sum()) if "drift_level" in report.columns else 0
            total_count = len(report)

            if high_count > 0:
                drift_summary = "High Shift Detected - Pipeline Investigation Recommended"
                summary_class = "high"
            elif mod_count > 0:
                drift_summary = "Moderate Shift Detected - Routine Monitoring Suggested"
                summary_class = "moderate"
            else:
                drift_summary = "Population Stable - Low Distribution Shift"
                summary_class = "low"

            stats = {
                "high_count": high_count,
                "mod_count": mod_count,
                "low_count": low_count,
                "total_count": total_count,
                "summary": drift_summary,
                "summary_class": summary_class,
                "source_file": Path(file_path).name,
            }

            return render_template("placement/monitoring.html", report=report, stats=stats, error=None)
        except Exception as e:
            return render_template("placement/monitoring.html", report=None, stats=None, error=str(e))

    return render_template("placement/monitoring.html", report=None, stats=None, error=None)


@placement_bp.route("/model_info")
@placement_bp.route("/model-info")
def model_info():
    try:
        artifact = get_artifact()
        return render_template("placement/model_info.html", artifact=artifact, error=None)
    except Exception as e:
        return render_template("placement/model_info.html", artifact=None, error=str(e))


@placement_bp.route("/retrain", methods=["GET", "POST"])
def retrain():
    try:
        train_placement_model()
        return redirect(url_for("placement.model_info"))
    except Exception as e:
        return redirect(url_for("placement.index"))
