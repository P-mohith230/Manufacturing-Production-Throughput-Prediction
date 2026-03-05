"""Feature engineering utilities for manufacturing throughput data."""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


NUMERIC_FEATURES = [
    "shift_duration",
    "machine_age",
    "equipment_utilization",
    "maintenance_frequency",
    "num_workers",
    "operator_skill_level",
    "raw_material_quality",
    "defect_rate",
    "temperature",
    "humidity",
    "production_line_speed",
    "downtime_hours",
    "batch_size",
    "energy_consumption",
    "quality_check_frequency",
    "rework_rate",
    "material_availability",
]

TARGET = "throughput"


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add engineered features to the manufacturing dataframe.

    Args:
        df: Raw manufacturing dataframe.

    Returns:
        DataFrame with additional engineered features.
    """
    df = df.copy()

    # Efficiency index
    df["efficiency_index"] = (
        df["equipment_utilization"]
        * df["raw_material_quality"]
        * df["material_availability"]
        * (1 - df["defect_rate"])
        * (1 - df["rework_rate"])
    )

    # Workforce productivity
    df["workforce_productivity"] = df["num_workers"] * df["operator_skill_level"]

    # Machine health score (newer + more maintenance = healthier)
    df["machine_health"] = (
        df["maintenance_frequency"] / (df["machine_age"] + 1)
    ).clip(upper=10)

    # Optimal temperature deviation (22°C is optimal)
    df["temp_deviation"] = np.abs(df["temperature"] - 22)

    # Optimal humidity deviation (50% is optimal)
    df["humidity_deviation"] = np.abs(df["humidity"] - 50)

    # Net productive time
    df["net_productive_time"] = df["shift_duration"] - df["downtime_hours"]
    df["net_productive_time"] = df["net_productive_time"].clip(lower=0)

    # Throughput rate potential
    df["line_time_product"] = df["production_line_speed"] * df["net_productive_time"]

    # Shift encoding (ordinal: morning=2, afternoon=1, night=0)
    shift_map = {"Morning": 2, "Afternoon": 1, "Night": 0}
    df["shift_encoded"] = df["shift"].map(shift_map)

    return df


def get_feature_columns(df: pd.DataFrame) -> list:
    """Return the list of feature columns to use for modelling.

    Args:
        df: Engineered dataframe.

    Returns:
        List of feature column names.
    """
    base = list(NUMERIC_FEATURES)
    engineered = [
        "efficiency_index",
        "workforce_productivity",
        "machine_health",
        "temp_deviation",
        "humidity_deviation",
        "net_productive_time",
        "line_time_product",
        "shift_encoded",
    ]
    return [c for c in base + engineered if c in df.columns]


def prepare_data(df: pd.DataFrame, scale: bool = False):
    """Prepare feature matrix X and target vector y.

    Args:
        df: Engineered dataframe.
        scale: If True, standardise X with StandardScaler.

    Returns:
        Tuple of (X, y) or (X, y, scaler) when scale=True.
    """
    feature_cols = get_feature_columns(df)
    X = df[feature_cols].copy()
    y = df[TARGET].copy()

    if scale:
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=feature_cols, index=X.index)
        return X_scaled, y, scaler

    return X, y
