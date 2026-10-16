"""
Population Stability Index (PSI) and Data Drift Monitoring Module.

Provides robust calculation of PSI for both numeric and categorical features
in the Student Placement prediction pipeline.
"""

from typing import Optional, Union, Dict, Any
import numpy as np
import pandas as pd


def calculate_psi(reference: Union[pd.Series, np.ndarray, list],
                  current: Union[pd.Series, np.ndarray, list],
                  bins: int = 10) -> Optional[float]:
    """
    Calculate Population Stability Index (PSI) for a continuous / numeric variable.

    PSI interpretation:
        PSI < 0.10: No significant population shift (stable)
        0.10 <= PSI < 0.25: Moderate shift; monitoring suggested
        PSI >= 0.25: High shift; action or model retraining required
    """
    reference = pd.to_numeric(pd.Series(reference), errors="coerce").dropna()
    current = pd.to_numeric(pd.Series(current), errors="coerce").dropna()

    if len(reference) == 0 or len(current) == 0:
        return None

    # Calculate percentile breaks from reference population
    breaks = np.unique(np.percentile(reference, np.linspace(0, 100, bins + 1)))
    if len(breaks) < 2:
        return 0.0

    # Expand boundary bins to cover values outside training reference
    breaks = breaks.copy()
    breaks[0] = -np.inf
    breaks[-1] = np.inf

    ref_count, _ = np.histogram(reference, bins=breaks)
    cur_count, _ = np.histogram(current, bins=breaks)

    def pct(x):
        x = x.astype(float)
        x = x / x.sum() if x.sum() > 0 else np.ones_like(x) / len(x)
        # Apply small epsilon to avoid log(0) and division by zero
        x = np.where(x == 0, 0.0001, x)
        return x / x.sum()

    r = pct(ref_count)
    c = pct(cur_count)

    psi_value = float(np.sum((c - r) * np.log(c / r)))
    return max(0.0, psi_value)


def calculate_categorical_psi(reference: Union[pd.Series, list],
                              current: Union[pd.Series, list]) -> Optional[float]:
    """
    Calculate Population Stability Index (PSI) for categorical variables
    such as gender, branch, college_tier, and volunteer_experience.
    """
    ref_series = pd.Series(reference).dropna().astype(str).str.strip()
    cur_series = pd.Series(current).dropna().astype(str).str.strip()

    if len(ref_series) == 0 or len(cur_series) == 0:
        return None

    all_categories = sorted(list(set(ref_series.unique()).union(set(cur_series.unique()))))
    if len(all_categories) <= 1:
        return 0.0

    ref_counts = ref_series.value_counts(normalize=True)
    cur_counts = cur_series.value_counts(normalize=True)

    r = np.array([ref_counts.get(cat, 0.0001) for cat in all_categories], dtype=float)
    c = np.array([cur_counts.get(cat, 0.0001) for cat in all_categories], dtype=float)

    r = r / r.sum()
    c = c / c.sum()

    psi_value = float(np.sum((c - r) * np.log(c / r)))
    return max(0.0, psi_value)


def interpret_psi(value: Optional[float]) -> str:
    """Interpret the calculated PSI value into standard MLOps status labels."""
    if value is None:
        return "Insufficient data"
    if value < 0.10:
        return "Low shift"
    if value < 0.25:
        return "Moderate shift"
    return "High shift - investigate"


def get_drift_level(value: Optional[float]) -> str:
    """Return css class/badge token for the drift status."""
    if value is None:
        return "unknown"
    if value < 0.10:
        return "low"
    if value < 0.25:
        return "moderate"
    return "high"


def calculate_drift_report(reference_df: pd.DataFrame,
                           current_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate a full drift report across all numeric and categorical features.

    Returns a DataFrame containing:
        - feature: Feature name
        - feature_type: 'Numeric' or 'Categorical'
        - psi: Population Stability Index (rounded to 4 decimal places)
        - status: Human-readable shift interpretation
        - drift_level: 'low', 'moderate', or 'high'
        - bar_width: Progress bar width representation (0 - 100%)
        - ref_baseline: Human-readable baseline statistic
        - cur_value: Human-readable current statistic
    """
    rows = []

    # Map column names case-insensitively for resilience
    cur_cols_map = {str(col).strip().lower(): col for col in current_df.columns}

    for col in reference_df.columns:
        col_key = str(col).strip().lower()
        if col_key not in cur_cols_map:
            continue

        actual_cur_col = cur_cols_map[col_key]
        ref_series = reference_df[col]
        cur_series = current_df[actual_cur_col]

        is_numeric = pd.api.types.is_numeric_dtype(ref_series)

        if is_numeric:
            feature_type = "Numeric"
            value = calculate_psi(ref_series, cur_series)

            r_clean = pd.to_numeric(ref_series, errors="coerce").dropna()
            c_clean = pd.to_numeric(cur_series, errors="coerce").dropna()
            r_mean = f"{r_clean.mean():.2f}" if len(r_clean) else "N/A"
            c_mean = f"{c_clean.mean():.2f}" if len(c_clean) else "N/A"
            ref_stat = f"Mean: {r_mean}"
            cur_stat = f"Mean: {c_mean}"
        else:
            feature_type = "Categorical"
            value = calculate_categorical_psi(ref_series, cur_series)

            r_mode = ref_series.mode().iloc[0] if not ref_series.empty else "N/A"
            c_mode = cur_series.mode().iloc[0] if not cur_series.empty else "N/A"
            ref_stat = f"Top: {r_mode}"
            cur_stat = f"Top: {c_mode}"

        rounded_psi = None if value is None else round(value, 4)
        status_text = interpret_psi(value)
        drift_lvl = get_drift_level(value)

        # Bar width capped at 100% (where 0.50 PSI maps to ~100% visual bar)
        bar_pct = 0
        if rounded_psi is not None:
            bar_pct = min(100, int((rounded_psi / 0.50) * 100))

        rows.append({
            "feature": col,
            "feature_type": feature_type,
            "psi": rounded_psi,
            "status": status_text,
            "drift_level": drift_lvl,
            "bar_width": bar_pct,
            "ref_baseline": ref_stat,
            "cur_value": cur_stat,
        })

    # Return DataFrame sorted by PSI descending (highest drift first)
    report_df = pd.DataFrame(rows)
    if not report_df.empty and "psi" in report_df.columns:
        report_df = report_df.sort_values(
            by=["psi"],
            ascending=False,
            na_position="last"
        ).reset_index(drop=True)

    return report_df
