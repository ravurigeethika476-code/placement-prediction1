import pandas as pd
from sklearn.preprocessing import LabelEncoder


# =========================================================
# MISSING VALUES SUMMARY
# =========================================================

def get_missing_values_summary(df):

    summary = pd.DataFrame()

    summary["Column"] = df.columns
    summary["Data Type"] = df.dtypes.astype(str).values
    summary["Missing Values"] = df.isnull().sum().values
    summary["Missing Percentage"] = (
        (df.isnull().sum() / len(df)) * 100
    ).round(2).values

    summary["Status"] = summary["Missing Values"].apply(
        lambda x: "Missing" if x > 0 else "No Missing"
    )

    return summary


# =========================================================
# FILL MISSING VALUES
# =========================================================

def fill_missing_values(df, column, method):

    if column not in df.columns:
        return df

    if method == "mean":

        if pd.api.types.is_numeric_dtype(df[column]):
            df[column] = df[column].fillna(
                df[column].mean()
            )

    elif method == "median":

        if pd.api.types.is_numeric_dtype(df[column]):
            df[column] = df[column].fillna(
                df[column].median()
            )

    elif method == "mode":

        mode = df[column].mode()

        if not mode.empty:
            df[column] = df[column].fillna(
                mode.iloc[0]
            )

    elif method == "zero":

        df[column] = df[column].fillna(0)

    elif method == "ffill":

        df[column] = df[column].ffill()

    elif method == "bfill":

        df[column] = df[column].bfill()

    elif method == "drop":

        df = df.dropna(subset=[column])

    return df


# =========================================================
# ENCODING
# =========================================================

def encode_column(df, column, method):

    if column not in df.columns:
        return df

    # Handle remaining missing categorical values
    df[column] = df[column].fillna("Unknown")

    # Label Encoding
    if method == "label":

        encoder = LabelEncoder()

        df[column] = encoder.fit_transform(
            df[column].astype(str)
        )

    # One Hot Encoding
    elif method == "onehot":

        df = pd.get_dummies(
            df,
            columns=[column],
            dtype=int
        )

    return df