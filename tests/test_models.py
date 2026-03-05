"""Tests for the manufacturing throughput prediction project."""

import numpy as np
import pandas as pd
import pytest

from src.data_generator import generate_manufacturing_data
from src.feature_engineering import (
    engineer_features,
    get_feature_columns,
    prepare_data,
)
from src.models import (
    compute_metrics,
    cross_validate_models,
    evaluate_models,
    get_feature_importance,
    predict_single,
    train_all_models,
    train_test_split_data,
)


# ──────────────────────────────────────────────────────────────────────────────
# Data Generator
# ──────────────────────────────────────────────────────────────────────────────

class TestDataGenerator:
    def test_returns_dataframe(self):
        df = generate_manufacturing_data(n_samples=100)
        assert isinstance(df, pd.DataFrame)

    def test_row_count(self):
        df = generate_manufacturing_data(n_samples=200)
        assert len(df) == 200

    def test_required_columns_present(self):
        df = generate_manufacturing_data(n_samples=50)
        expected = [
            "date", "shift", "shift_duration", "machine_age",
            "equipment_utilization", "num_workers", "throughput",
        ]
        for col in expected:
            assert col in df.columns, f"Column '{col}' missing"

    def test_throughput_positive(self):
        df = generate_manufacturing_data(n_samples=200)
        assert (df["throughput"] > 0).all()

    def test_reproducible_with_same_seed(self):
        df1 = generate_manufacturing_data(n_samples=100, random_state=0)
        df2 = generate_manufacturing_data(n_samples=100, random_state=0)
        pd.testing.assert_frame_equal(df1, df2)

    def test_different_seeds_differ(self):
        df1 = generate_manufacturing_data(n_samples=100, random_state=0)
        df2 = generate_manufacturing_data(n_samples=100, random_state=99)
        assert not df1["throughput"].equals(df2["throughput"])

    def test_shift_values(self):
        df = generate_manufacturing_data(n_samples=300)
        assert set(df["shift"].unique()).issubset({"Morning", "Afternoon", "Night"})


# ──────────────────────────────────────────────────────────────────────────────
# Feature Engineering
# ──────────────────────────────────────────────────────────────────────────────

class TestFeatureEngineering:
    @pytest.fixture
    def raw_df(self):
        return generate_manufacturing_data(n_samples=200)

    def test_engineer_features_adds_columns(self, raw_df):
        engineered = engineer_features(raw_df)
        new_cols = [
            "efficiency_index", "workforce_productivity", "machine_health",
            "temp_deviation", "humidity_deviation", "net_productive_time",
            "line_time_product", "shift_encoded",
        ]
        for col in new_cols:
            assert col in engineered.columns, f"Engineered column '{col}' missing"

    def test_does_not_modify_original(self, raw_df):
        original_cols = list(raw_df.columns)
        _ = engineer_features(raw_df)
        assert list(raw_df.columns) == original_cols

    def test_efficiency_index_bounds(self, raw_df):
        eng = engineer_features(raw_df)
        assert (eng["efficiency_index"] >= 0).all()
        assert (eng["efficiency_index"] <= 1).all()

    def test_net_productive_time_nonneg(self, raw_df):
        eng = engineer_features(raw_df)
        assert (eng["net_productive_time"] >= 0).all()

    def test_shift_encoded_values(self, raw_df):
        eng = engineer_features(raw_df)
        assert set(eng["shift_encoded"].unique()).issubset({0, 1, 2})

    def test_get_feature_columns_returns_list(self, raw_df):
        eng = engineer_features(raw_df)
        cols = get_feature_columns(eng)
        assert isinstance(cols, list)
        assert len(cols) > 0

    def test_prepare_data_shapes(self, raw_df):
        eng = engineer_features(raw_df)
        X, y = prepare_data(eng)
        assert X.shape[0] == len(eng)
        assert y.shape[0] == len(eng)
        assert "throughput" not in X.columns

    def test_prepare_data_scaled(self, raw_df):
        eng = engineer_features(raw_df)
        X_s, y, scaler = prepare_data(eng, scale=True)
        # Scaled features should have near-zero mean
        assert abs(X_s.mean().mean()) < 0.1


# ──────────────────────────────────────────────────────────────────────────────
# Models
# ──────────────────────────────────────────────────────────────────────────────

class TestModels:
    @pytest.fixture(scope="class")
    def data(self):
        df = generate_manufacturing_data(n_samples=500)
        eng = engineer_features(df)
        X, y = prepare_data(eng)
        X_train, X_test, y_train, y_test = train_test_split_data(X, y)
        fitted = train_all_models(X_train, y_train)
        return fitted, X_train, X_test, y_train, y_test

    def test_four_models_trained(self, data):
        fitted, *_ = data
        assert len(fitted) == 4

    def test_model_names(self, data):
        fitted, *_ = data
        expected_names = {"Linear Regression", "Decision Tree", "Random Forest", "Gradient Boosting"}
        assert set(fitted.keys()) == expected_names

    def test_models_have_predict(self, data):
        fitted, _, X_test, _, _ = data
        for name, model in fitted.items():
            preds = model.predict(X_test)
            assert len(preds) == len(X_test), f"{name}: wrong prediction length"

    def test_predictions_positive(self, data):
        fitted, _, X_test, _, _ = data
        for name, model in fitted.items():
            preds = model.predict(X_test)
            assert (preds > 0).all(), f"{name}: non-positive predictions found"

    def test_evaluate_models_returns_dataframe(self, data):
        fitted, _, X_test, _, y_test = data
        metrics = evaluate_models(fitted, X_test, y_test)
        assert isinstance(metrics, pd.DataFrame)
        assert list(metrics.columns) == ["R2", "RMSE", "MAE", "MAPE"]
        assert len(metrics) == 4

    def test_r2_reasonable(self, data):
        fitted, _, X_test, _, y_test = data
        metrics = evaluate_models(fitted, X_test, y_test)
        # All models should achieve at least R² > 0.5 on this dataset
        assert (metrics["R2"] > 0.5).all(), f"Low R² detected:\n{metrics}"

    def test_compute_metrics_keys(self):
        y_true = np.array([100.0, 200.0, 300.0])
        y_pred = np.array([110.0, 195.0, 290.0])
        result = compute_metrics(y_true, y_pred)
        assert set(result.keys()) == {"R2", "RMSE", "MAE", "MAPE"}

    def test_compute_metrics_perfect(self):
        y = np.array([100.0, 200.0, 300.0])
        result = compute_metrics(y, y)
        assert result["R2"] == 1.0
        assert result["RMSE"] == 0.0
        assert result["MAE"] == 0.0
        assert result["MAPE"] == 0.0

    def test_cross_validate_models(self, data):
        fitted, X_train, X_test, y_train, y_test = data
        X_full = pd.concat([X_train, X_test])
        y_full = pd.concat([y_train, y_test])
        cv_df = cross_validate_models(fitted, X_full, y_full, cv=3)
        assert isinstance(cv_df, pd.DataFrame)
        assert "CV_R2_Mean" in cv_df.columns

    def test_feature_importance_tree_models(self, data):
        fitted, X_train, *_ = data
        feature_names = list(X_train.columns)
        importances = get_feature_importance(fitted, feature_names)
        assert "Random Forest" in importances
        assert "Gradient Boosting" in importances
        assert "Linear Regression" not in importances  # no feature_importances_ attr

    def test_feature_importance_sums_to_one(self, data):
        fitted, X_train, *_ = data
        feature_names = list(X_train.columns)
        importances = get_feature_importance(fitted, feature_names)
        for name, imp in importances.items():
            assert abs(imp.sum() - 1.0) < 1e-6, f"{name} importances don't sum to 1"

    def test_predict_single(self, data):
        fitted, X_train, *_ = data
        feature_names = list(X_train.columns)
        feature_values = {f: float(X_train[f].mean()) for f in feature_names}
        for name, model in fitted.items():
            result = predict_single(model, feature_values, feature_names)
            assert isinstance(result, float)
            assert result > 0

    def test_train_test_split_sizes(self):
        df = generate_manufacturing_data(n_samples=200)
        eng = engineer_features(df)
        X, y = prepare_data(eng)
        X_train, X_test, y_train, y_test = train_test_split_data(X, y, test_size=0.2)
        assert len(X_test) == pytest.approx(40, abs=1)
        assert len(X_train) + len(X_test) == len(X)
