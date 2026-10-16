from pathlib import Path
from typing import Union
import pandas as pd
from monitoring.psi import calculate_drift_report

BASE = Path(__file__).resolve().parent.parent
REFERENCE_DIR = BASE / "reference_data"
REFERENCE = REFERENCE_DIR / "training_reference.csv"


def generate_report(current_source: Union[str, Path, pd.DataFrame], max_samples: int = 30000) -> pd.DataFrame:
    """
    Generate PSI data drift report comparing current dataset against
    the baseline training reference dataset.

    Args:
        current_source: Path to .csv/.xlsx/.xls file or a pandas DataFrame.
        max_samples: Maximum number of rows to evaluate for high-speed robust calculation.

    Returns:
        pd.DataFrame containing feature drift metrics and status.
    """
    if not REFERENCE.exists():
        # Auto-initialize model and reference data if not yet created
        from placement.model import get_artifact
        get_artifact()

    if not REFERENCE.exists():
        raise FileNotFoundError(
            "Baseline reference data not found at reference_data/training_reference.csv. "
            "Please train or retrain the model first."
        )

    ref = pd.read_csv(REFERENCE)

    if isinstance(current_source, pd.DataFrame):
        cur = current_source.copy()
    else:
        file_path = Path(current_source)
        if not file_path.exists():
            raise FileNotFoundError(f"Uploaded file not found: {file_path}")

        suffix = file_path.suffix.lower()
        if suffix in [".xlsx", ".xls"]:
            cur = pd.read_excel(file_path)
        else:
            cur = pd.read_csv(file_path)

    # Normalize column comparison (case-insensitive & stripped)
    ref_col_map = {c.strip().lower(): c for c in ref.columns}
    cur_col_map = {str(c).strip().lower(): c for c in cur.columns}

    missing = [ref_col_map[k] for k in ref_col_map if k not in cur_col_map]
    if missing:
        if len(missing) == len(ref.columns):
            raise ValueError(
                f"Current dataset contains none of the required placement features. "
                f"Required features include: {', '.join(list(ref.columns)[:8])}..."
            )
        raise ValueError(
            f"Current data is missing {len(missing)} required feature(s): {', '.join(missing)}"
        )

    # Standardize current column names to match reference casing
    rename_dict = {cur_col_map[k]: ref_col_map[k] for k in ref_col_map if k in cur_col_map}
    cur = cur.rename(columns=rename_dict)

    # Fast sampling if dataset is massive (maintains robust distribution accuracy)
    if max_samples and len(cur) > max_samples:
        cur = cur.sample(n=max_samples, random_state=42)

    return calculate_drift_report(ref, cur)
