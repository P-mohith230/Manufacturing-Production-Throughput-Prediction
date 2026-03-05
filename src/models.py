"""ML model training, evaluation and comparison for throughput prediction."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.tree import DecisionTreeRegressor


MODEL_DEFINITIONS: dict[str, object] = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(max_depth=8, random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42),
}


def train_test_split_data(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42):
    """Split data into train and test sets.

    Args:
        X: Feature matrix.
        y: Target series.
        test_size: Fraction of data for testing.
        random_state: Random seed.

    Returns:
        Tuple of (X_train, X_test, y_train, y_test).
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute regression evaluation metrics.

    Args:
        y_true: True target values.
        y_pred: Predicted target values.

    Returns:
        Dictionary with R2, RMSE, MAE and MAPE.
    """
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    # MAPE – guard against zero-valued actuals
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100 if mask.any() else np.nan
    return {"R2": round(r2, 4), "RMSE": round(rmse, 2), "MAE": round(mae, 2), "MAPE": round(mape, 2)}


def train_all_models(X_train: pd.DataFrame, y_train: pd.Series) -> dict:
    """Fit all four models on training data.

    Args:
        X_train: Training feature matrix.
        y_train: Training target series.

    Returns:
        Dictionary mapping model name -> fitted estimator.
    """
    fitted: dict = {}
    for name, model in MODEL_DEFINITIONS.items():
        import copy
        m = copy.deepcopy(model)
        m.fit(X_train, y_train)
        fitted[name] = m
    return fitted


def evaluate_models(
    fitted_models: dict,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> pd.DataFrame:
    """Evaluate fitted models on the test set.

    Args:
        fitted_models: Dictionary of name -> fitted estimator.
        X_test: Test feature matrix.
        y_test: Test target series.

    Returns:
        DataFrame with one row per model and metric columns.
    """
    rows = []
    for name, model in fitted_models.items():
        y_pred = model.predict(X_test)
        metrics = compute_metrics(y_test.values, y_pred)
        metrics["Model"] = name
        rows.append(metrics)
    results = pd.DataFrame(rows).set_index("Model")
    return results[["R2", "RMSE", "MAE", "MAPE"]]


def cross_validate_models(
    fitted_models: dict,
    X: pd.DataFrame,
    y: pd.Series,
    cv: int = 5,
) -> pd.DataFrame:
    """Run k-fold cross-validation for each model.

    Args:
        fitted_models: Dictionary of name -> fitted estimator (used for type reference only).
        X: Full feature matrix.
        y: Full target series.
        cv: Number of folds.

    Returns:
        DataFrame with mean and std of CV R² per model.
    """
    import copy

    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    rows = []
    for name, model in fitted_models.items():
        m = copy.deepcopy(model)
        scores = cross_val_score(m, X, y, cv=kf, scoring="r2", n_jobs=-1)
        rows.append({"Model": name, "CV_R2_Mean": round(scores.mean(), 4), "CV_R2_Std": round(scores.std(), 4)})
    return pd.DataFrame(rows).set_index("Model")


def get_feature_importance(fitted_models: dict, feature_names: list) -> dict[str, pd.Series]:
    """Extract feature importances for tree-based models.

    Args:
        fitted_models: Dictionary of name -> fitted estimator.
        feature_names: List of feature column names.

    Returns:
        Dictionary mapping model name -> Series of feature importances.
    """
    importances: dict = {}
    for name, model in fitted_models.items():
        if hasattr(model, "feature_importances_"):
            imp = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False)
            importances[name] = imp
    return importances


def predict_single(model, feature_values: dict, feature_names: list) -> float:
    """Generate a single throughput prediction.

    Args:
        model: Fitted estimator.
        feature_values: Dictionary of feature name -> value.
        feature_names: Ordered list of feature names expected by the model.

    Returns:
        Predicted throughput value.
    """
    X = pd.DataFrame([[feature_values[f] for f in feature_names]], columns=feature_names)
    return float(model.predict(X)[0])
