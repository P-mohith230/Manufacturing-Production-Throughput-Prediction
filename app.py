"""Manufacturing Production Throughput Prediction — Streamlit Dashboard."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from sklearn.model_selection import learning_curve

from src.data_generator import generate_manufacturing_data
from src.feature_engineering import engineer_features, get_feature_columns, prepare_data
from src.models import (
    cross_validate_models,
    evaluate_models,
    get_feature_importance,
    predict_single,
    train_all_models,
    train_test_split_data,
)

# ──────────────────────────────────────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Manufacturing Throughput Prediction",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────────────
# Caching helpers
# ──────────────────────────────────────────────────────────────────────────────

@st.cache_data
def load_data(n_samples: int = 2000) -> pd.DataFrame:
    raw = generate_manufacturing_data(n_samples=n_samples)
    return engineer_features(raw)


@st.cache_data
def get_trained_models(n_samples: int = 2000):
    df = load_data(n_samples)
    X, y = prepare_data(df)
    X_train, X_test, y_train, y_test = train_test_split_data(X, y)
    fitted = train_all_models(X_train, y_train)
    metrics = evaluate_models(fitted, X_test, y_test)
    cv_metrics = cross_validate_models(fitted, X, y)
    feature_names = get_feature_columns(df)
    importances = get_feature_importance(fitted, feature_names)
    predictions = {name: model.predict(X_test) for name, model in fitted.items()}
    return fitted, metrics, cv_metrics, importances, X_test, y_test, predictions, feature_names


# ──────────────────────────────────────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────────────────────────────────────
st.sidebar.image(
    "https://img.icons8.com/color/96/000000/factory.png",
    width=80,
)
st.sidebar.title("🏭 Manufacturing Throughput Prediction")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate",
    [
        "📊 Data Overview",
        "🔍 Feature Analysis",
        "🤖 Model Performance",
        "📈 Model Insights",
        "🎯 Real-Time Predictor",
    ],
)

n_samples = st.sidebar.slider("Dataset Size", 500, 5000, 2000, 500)
st.sidebar.markdown("---")
st.sidebar.markdown("**Models Used**")
st.sidebar.markdown("• Linear Regression\n• Decision Tree\n• Random Forest\n• Gradient Boosting")

# ──────────────────────────────────────────────────────────────────────────────
# Load data & models
# ──────────────────────────────────────────────────────────────────────────────
df = load_data(n_samples)
fitted, metrics, cv_metrics, importances, X_test, y_test, predictions, feature_names = get_trained_models(n_samples)

PALETTE = px.colors.qualitative.Set2
MODEL_COLORS = {
    "Linear Regression": PALETTE[0],
    "Decision Tree": PALETTE[1],
    "Random Forest": PALETTE[2],
    "Gradient Boosting": PALETTE[3],
}

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Data Overview
# ══════════════════════════════════════════════════════════════════════════════
if page == "📊 Data Overview":
    st.title("📊 Manufacturing Data Overview")
    st.markdown("Explore the synthetic manufacturing dataset used to train and evaluate the prediction models.")

    # ── KPI Cards ──
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Records", f"{len(df):,}")
    col2.metric("Avg Throughput", f"{df['throughput'].mean():.0f} units")
    col3.metric("Max Throughput", f"{df['throughput'].max():.0f} units")
    col4.metric("Min Throughput", f"{df['throughput'].min():.0f} units")

    st.markdown("---")

    # ── VIZ 1: Throughput distribution ──
    col_l, col_r = st.columns(2)
    with col_l:
        st.subheader("1. Throughput Distribution")
        fig = px.histogram(
            df, x="throughput", nbins=50, color_discrete_sequence=["#636EFA"],
            labels={"throughput": "Throughput (units/shift)"},
            title="Distribution of Production Throughput",
        )
        fig.update_layout(bargap=0.05, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    # ── VIZ 2: Throughput by shift ──
    with col_r:
        st.subheader("2. Throughput by Shift")
        fig = px.box(
            df, x="shift", y="throughput", color="shift",
            color_discrete_sequence=PALETTE,
            title="Throughput Distribution by Shift",
            labels={"throughput": "Throughput (units/shift)", "shift": "Shift"},
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    # ── VIZ 3: Time series ──
    st.subheader("3. Throughput Over Time (rolling average)")
    df_ts = df[["date", "throughput"]].copy()
    df_ts["rolling_avg"] = df_ts["throughput"].rolling(24).mean()
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_ts["date"], y=df_ts["throughput"],
        mode="lines", name="Actual", line=dict(color="lightblue", width=0.8), opacity=0.5,
    ))
    fig.add_trace(go.Scatter(
        x=df_ts["date"], y=df_ts["rolling_avg"],
        mode="lines", name="24-hr Rolling Avg", line=dict(color="#EF553B", width=2),
    ))
    fig.update_layout(title="Throughput Time Series", xaxis_title="Date", yaxis_title="Throughput")
    st.plotly_chart(fig, use_container_width=True)

    # ── VIZ 4: Correlation heatmap ──
    st.subheader("4. Feature Correlation Heatmap")
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    corr = df[numeric_cols].corr()
    fig = px.imshow(
        corr, text_auto=".2f", aspect="auto",
        color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
        title="Pearson Correlation Matrix",
    )
    fig.update_layout(height=700)
    st.plotly_chart(fig, use_container_width=True)

    # ── Raw data preview ──
    st.subheader("Raw Dataset Sample")
    st.dataframe(df.head(100), use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Feature Analysis
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Feature Analysis":
    st.title("🔍 Feature Analysis")
    st.markdown("Deep-dive into how individual features relate to throughput.")

    base_features = [
        "equipment_utilization", "machine_age", "operator_skill_level",
        "raw_material_quality", "defect_rate", "production_line_speed",
        "num_workers", "downtime_hours", "maintenance_frequency", "batch_size",
    ]

    # ── VIZ 5: Scatter matrix ──
    st.subheader("5. Scatter Matrix (Selected Features vs Throughput)")
    scatter_features = st.multiselect(
        "Choose features for scatter matrix:",
        options=base_features,
        default=["equipment_utilization", "production_line_speed", "defect_rate", "operator_skill_level"],
    )
    if len(scatter_features) >= 2:
        fig = px.scatter_matrix(
            df, dimensions=scatter_features + ["throughput"],
            color="shift", color_discrete_sequence=PALETTE,
            title="Pairwise Scatter Matrix",
            labels={c: c.replace("_", " ").title() for c in scatter_features + ["throughput"]},
        )
        fig.update_traces(diagonal_visible=False, showupperhalf=False)
        st.plotly_chart(fig, use_container_width=True)

    col_l, col_r = st.columns(2)

    # ── VIZ 6: Feature vs throughput (user-selected) ──
    with col_l:
        st.subheader("6. Feature vs Throughput Scatter")
        sel_feature = st.selectbox("Select feature:", base_features, index=0, key="feat_scatter")
        fig = px.scatter(
            df, x=sel_feature, y="throughput", color="shift",
            color_discrete_sequence=PALETTE, opacity=0.6, trendline="ols",
            title=f"{sel_feature.replace('_', ' ').title()} vs Throughput",
            labels={"throughput": "Throughput", sel_feature: sel_feature.replace("_", " ").title()},
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── VIZ 7: Feature distributions (violin) ──
    with col_r:
        st.subheader("7. Feature Distributions by Shift (Violin)")
        sel_violin = st.selectbox("Select feature:", base_features, index=2, key="feat_violin")
        fig = px.violin(
            df, x="shift", y=sel_violin, color="shift",
            box=True, points="outliers", color_discrete_sequence=PALETTE,
            title=f"Distribution of {sel_violin.replace('_', ' ').title()} by Shift",
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    # ── VIZ 8: Parallel coordinates ──
    st.subheader("8. Parallel Coordinates Plot")
    parallel_features = [
        "equipment_utilization", "production_line_speed", "defect_rate",
        "operator_skill_level", "machine_age", "throughput",
    ]
    df_para = df[parallel_features + ["shift_encoded"]].copy()
    fig = px.parallel_coordinates(
        df_para, color="throughput",
        color_continuous_scale=px.colors.sequential.Viridis,
        dimensions=parallel_features,
        title="Parallel Coordinates — Multi-feature Throughput View",
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── VIZ 9: 3D scatter ──
    st.subheader("9. 3D Scatter Plot")
    col_x = st.selectbox("X axis:", base_features, index=5, key="3d_x")
    col_y = st.selectbox("Y axis:", base_features, index=0, key="3d_y")
    fig = px.scatter_3d(
        df.sample(500, random_state=0), x=col_x, y=col_y, z="throughput",
        color="shift", color_discrete_sequence=PALETTE, opacity=0.7,
        title=f"3D: {col_x} × {col_y} → Throughput",
        labels={col_x: col_x.replace("_", " ").title(),
                col_y: col_y.replace("_", " ").title(),
                "throughput": "Throughput"},
    )
    st.plotly_chart(fig, use_container_width=True)

    col_l2, col_r2 = st.columns(2)

    # ── VIZ 10: Defect rate vs throughput heatmap (2D histogram) ──
    with col_l2:
        st.subheader("10. Defect Rate vs Throughput (Density)")
        fig = px.density_heatmap(
            df, x="defect_rate", y="throughput", nbinsx=30, nbinsy=30,
            color_continuous_scale="Blues",
            title="Defect Rate vs Throughput Density",
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── VIZ 11: Monthly aggregated throughput bar ──
    with col_r2:
        st.subheader("11. Monthly Average Throughput")
        df_monthly = df.copy()
        df_monthly["month"] = df_monthly["date"].dt.to_period("M").astype(str)
        monthly_avg = df_monthly.groupby("month")["throughput"].mean().reset_index()
        fig = px.bar(
            monthly_avg, x="month", y="throughput",
            color="throughput", color_continuous_scale="Teal",
            title="Average Monthly Throughput",
            labels={"throughput": "Avg Throughput", "month": "Month"},
        )
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Model Performance
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🤖 Model Performance":
    st.title("🤖 Model Performance Comparison")
    st.markdown("Evaluate and compare the four regression models on the test set.")

    # ── KPI Cards (Best Model) ──
    best_model = metrics["R2"].idxmax()
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Best Model", best_model)
    col2.metric("Best R²", f"{metrics.loc[best_model, 'R2']:.4f}")
    col3.metric("Best RMSE", f"{metrics.loc[best_model, 'RMSE']:.1f}")
    col4.metric("Best MAE", f"{metrics.loc[best_model, 'MAE']:.1f}")

    st.markdown("---")
    st.subheader("Test-Set Metrics Table")
    st.dataframe(metrics.style.highlight_max(axis=0, subset=["R2"], color="#d4edda")
                              .highlight_min(axis=0, subset=["RMSE", "MAE", "MAPE"], color="#d4edda"),
                 use_container_width=True)

    col_l, col_r = st.columns(2)

    # ── VIZ 12: R² bar chart ──
    with col_l:
        st.subheader("12. R² Score Comparison")
        fig = px.bar(
            metrics.reset_index(), x="Model", y="R2",
            color="Model", color_discrete_map=MODEL_COLORS,
            title="R² Score by Model", text_auto=".4f",
        )
        fig.update_layout(showlegend=False, yaxis_range=[0, 1])
        st.plotly_chart(fig, use_container_width=True)

    # ── VIZ 13: RMSE & MAE grouped bar ──
    with col_r:
        st.subheader("13. RMSE & MAE Comparison")
        fig = go.Figure()
        for metric_name, color in [("RMSE", "#EF553B"), ("MAE", "#00CC96")]:
            fig.add_trace(go.Bar(
                name=metric_name,
                x=metrics.index.tolist(),
                y=metrics[metric_name].tolist(),
                marker_color=color,
                text=[f"{v:.1f}" for v in metrics[metric_name]],
                textposition="outside",
            ))
        fig.update_layout(barmode="group", title="RMSE and MAE by Model",
                          yaxis_title="Error (units)", xaxis_title="Model")
        st.plotly_chart(fig, use_container_width=True)

    # ── VIZ 14: Actual vs Predicted (all models) ──
    st.subheader("14. Actual vs Predicted — All Models")
    fig = make_subplots(rows=2, cols=2, subplot_titles=list(predictions.keys()))
    positions = [(1, 1), (1, 2), (2, 1), (2, 2)]
    for i, (name, y_pred) in enumerate(predictions.items()):
        row, col = positions[i]
        fig.add_trace(go.Scatter(
            x=y_test.values.tolist(), y=y_pred.tolist(),
            mode="markers", name=name, marker=dict(color=MODEL_COLORS[name], opacity=0.5, size=4),
            showlegend=True,
        ), row=row, col=col)
        line_min = float(min(y_test.min(), y_pred.min()))
        line_max = float(max(y_test.max(), y_pred.max()))
        fig.add_trace(go.Scatter(
            x=[line_min, line_max], y=[line_min, line_max],
            mode="lines", line=dict(dash="dash", color="black", width=1),
            showlegend=False,
        ), row=row, col=col)
    fig.update_layout(height=700, title_text="Actual vs Predicted Throughput")
    st.plotly_chart(fig, use_container_width=True)

    # ── VIZ 15: Residuals distribution ──
    st.subheader("15. Residuals Distribution")
    sel_model = st.selectbox("Select model for residuals:", list(predictions.keys()))
    residuals = y_test.values - predictions[sel_model]
    fig = make_subplots(rows=1, cols=2, subplot_titles=["Residuals Histogram", "Residuals vs Predicted"])
    fig.add_trace(go.Histogram(x=residuals, nbinsx=40, name="Residuals",
                               marker_color=MODEL_COLORS[sel_model], opacity=0.75), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=predictions[sel_model].tolist(), y=residuals.tolist(),
        mode="markers", marker=dict(color=MODEL_COLORS[sel_model], opacity=0.5, size=4),
        name="Residuals vs Predicted",
    ), row=1, col=2)
    fig.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=2)
    fig.update_layout(title_text=f"Residual Analysis — {sel_model}", showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    # ── VIZ 16: Cross-validation scores ──
    st.subheader("16. Cross-Validation R² Scores")
    cv_df = cv_metrics.reset_index()
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=cv_df["Model"], y=cv_df["CV_R2_Mean"],
        error_y=dict(type="data", array=cv_df["CV_R2_Std"].tolist(), visible=True),
        marker_color=list(MODEL_COLORS.values()),
        text=[f"{v:.4f}" for v in cv_df["CV_R2_Mean"]],
        textposition="outside",
    ))
    fig.update_layout(title="5-Fold Cross-Validation R² (mean ± std)",
                      yaxis_title="R²", xaxis_title="Model", yaxis_range=[0, 1])
    st.plotly_chart(fig, use_container_width=True)

    # ── VIZ 17: Radar chart ──
    st.subheader("17. Model Performance Radar Chart")
    radar_metrics = ["R2", "RMSE", "MAE", "MAPE"]
    # Normalise to 0-1 (higher R2 better; lower error better → invert)
    norm = metrics.copy()
    norm["RMSE"] = 1 - (norm["RMSE"] - norm["RMSE"].min()) / (norm["RMSE"].max() - norm["RMSE"].min() + 1e-9)
    norm["MAE"]  = 1 - (norm["MAE"]  - norm["MAE"].min())  / (norm["MAE"].max()  - norm["MAE"].min()  + 1e-9)
    norm["MAPE"] = 1 - (norm["MAPE"] - norm["MAPE"].min()) / (norm["MAPE"].max() - norm["MAPE"].min() + 1e-9)
    categories = ["R² Score", "RMSE (inv)", "MAE (inv)", "MAPE (inv)"]
    fig = go.Figure()
    for name in norm.index:
        vals = norm.loc[name, radar_metrics].tolist()
        fig.add_trace(go.Scatterpolar(
            r=vals + [vals[0]], theta=categories + [categories[0]],
            fill="toself", name=name, marker_color=MODEL_COLORS[name],
        ))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                      title="Model Performance Radar (normalised, higher=better)")
    st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Model Insights
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📈 Model Insights":
    st.title("📈 Model Insights & Feature Importance")

    # ── VIZ 18: Feature importance (Random Forest) ──
    st.subheader("18. Feature Importance — Random Forest")
    if "Random Forest" in importances:
        imp_rf = importances["Random Forest"].reset_index()
        imp_rf.columns = ["Feature", "Importance"]
        fig = px.bar(
            imp_rf.head(15), x="Importance", y="Feature", orientation="h",
            color="Importance", color_continuous_scale="Greens",
            title="Top 15 Feature Importances (Random Forest)",
        )
        fig.update_layout(yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, use_container_width=True)

    # ── VIZ 19: Feature importance (Gradient Boosting) ──
    st.subheader("19. Feature Importance — Gradient Boosting")
    if "Gradient Boosting" in importances:
        imp_gb = importances["Gradient Boosting"].reset_index()
        imp_gb.columns = ["Feature", "Importance"]
        fig = px.bar(
            imp_gb.head(15), x="Importance", y="Feature", orientation="h",
            color="Importance", color_continuous_scale="Oranges",
            title="Top 15 Feature Importances (Gradient Boosting)",
        )
        fig.update_layout(yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, use_container_width=True)

    # ── VIZ 20: Importance comparison side-by-side ──
    st.subheader("20. Feature Importance Comparison (RF vs GB)")
    if "Random Forest" in importances and "Gradient Boosting" in importances:
        imp_compare = pd.DataFrame({
            "Random Forest": importances["Random Forest"],
            "Gradient Boosting": importances["Gradient Boosting"],
        }).fillna(0).sort_values("Random Forest", ascending=False).head(12)
        fig = go.Figure()
        for model_name, color in [("Random Forest", MODEL_COLORS["Random Forest"]),
                                   ("Gradient Boosting", MODEL_COLORS["Gradient Boosting"])]:
            fig.add_trace(go.Bar(
                name=model_name,
                x=imp_compare.index.tolist(),
                y=imp_compare[model_name].tolist(),
                marker_color=color,
            ))
        fig.update_layout(barmode="group", title="Feature Importance: RF vs GB",
                          xaxis_tickangle=-30, yaxis_title="Importance")
        st.plotly_chart(fig, use_container_width=True)

    # ── VIZ 21: Learning curve ──
    st.subheader("21. Learning Curve")
    lc_model_name = st.selectbox("Select model:", list(fitted.keys()), key="lc_model")
    import copy
    lc_model = copy.deepcopy(fitted[lc_model_name])
    X_full, y_full = prepare_data(load_data(n_samples))
    with st.spinner("Computing learning curve…"):
        train_sizes, train_scores, val_scores = learning_curve(
            lc_model, X_full, y_full,
            train_sizes=np.linspace(0.1, 1.0, 8),
            cv=3, scoring="r2", n_jobs=-1,
        )
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=train_sizes.tolist(), y=train_scores.mean(axis=1).tolist(),
        mode="lines+markers", name="Train R²", line=dict(color="#636EFA"),
        error_y=dict(type="data", array=train_scores.std(axis=1).tolist(), visible=True),
    ))
    fig.add_trace(go.Scatter(
        x=train_sizes.tolist(), y=val_scores.mean(axis=1).tolist(),
        mode="lines+markers", name="Validation R²", line=dict(color="#EF553B"),
        error_y=dict(type="data", array=val_scores.std(axis=1).tolist(), visible=True),
    ))
    fig.update_layout(title=f"Learning Curve — {lc_model_name}",
                      xaxis_title="Training Set Size", yaxis_title="R² Score")
    st.plotly_chart(fig, use_container_width=True)

    # ── VIZ 22: Predicted throughput distribution ──
    st.subheader("22. Predicted vs Actual Throughput Distribution")
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=y_test.values.tolist(), name="Actual", opacity=0.6,
                               marker_color="steelblue", nbinsx=40))
    for name, y_pred in predictions.items():
        fig.add_trace(go.Histogram(x=y_pred.tolist(), name=f"Pred — {name}",
                                   opacity=0.4, nbinsx=40))
    fig.update_layout(barmode="overlay",
                      title="Predicted vs Actual Throughput Distributions",
                      xaxis_title="Throughput", yaxis_title="Count")
    st.plotly_chart(fig, use_container_width=True)

    # ── VIZ 23: MAPE gauge ──
    st.subheader("23. Model MAPE Gauge")
    sel_gauge = st.selectbox("Select model:", list(metrics.index), key="gauge_model")
    mape_val = float(metrics.loc[sel_gauge, "MAPE"])
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=mape_val,
        delta={"reference": 10, "increasing": {"color": "red"}, "decreasing": {"color": "green"}},
        gauge={
            "axis": {"range": [0, 30]},
            "bar": {"color": MODEL_COLORS[sel_gauge]},
            "steps": [
                {"range": [0, 5], "color": "lightgreen"},
                {"range": [5, 15], "color": "lightyellow"},
                {"range": [15, 30], "color": "lightcoral"},
            ],
            "threshold": {"line": {"color": "red", "width": 4}, "thickness": 0.75, "value": 10},
        },
        title={"text": f"MAPE (%) — {sel_gauge}"},
    ))
    st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Real-Time Predictor
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🎯 Real-Time Predictor":
    st.title("🎯 Real-Time Throughput Predictor")
    st.markdown("Adjust the sliders to simulate production conditions and get an instant throughput prediction.")

    pred_model_name = st.selectbox("Choose prediction model:", list(fitted.keys()))
    pred_model = fitted[pred_model_name]

    st.markdown("---")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("⚙️ Equipment")
        machine_age = st.slider("Machine Age (years)", 0.5, 15.0, 5.0, 0.5)
        equipment_utilization = st.slider("Equipment Utilization", 0.50, 1.0, 0.80, 0.01)
        maintenance_frequency = st.slider("Maintenance Freq (times/yr)", 1, 12, 6)
        production_line_speed = st.slider("Line Speed (units/min)", 50.0, 150.0, 100.0, 1.0)
        energy_consumption = st.slider("Energy Consumption (kWh)", 100.0, 500.0, 300.0, 10.0)

    with col2:
        st.subheader("👷 Workforce & Process")
        num_workers = st.slider("Number of Workers", 5, 30, 15)
        operator_skill_level = st.slider("Operator Skill Level (1-10)", 1.0, 10.0, 7.0, 0.5)
        shift_duration = st.slider("Shift Duration (hrs)", 6.0, 10.0, 8.0, 0.5)
        downtime_hours = st.slider("Downtime (hrs)", 0.0, 3.0, 0.5, 0.1)
        quality_check_frequency = st.slider("Quality Checks / Shift", 1, 10, 5)

    with col3:
        st.subheader("📦 Material & Environment")
        raw_material_quality = st.slider("Raw Material Quality", 0.5, 1.0, 0.85, 0.01)
        defect_rate = st.slider("Defect Rate", 0.01, 0.15, 0.04, 0.01)
        rework_rate = st.slider("Rework Rate", 0.0, 0.10, 0.02, 0.01)
        material_availability = st.slider("Material Availability", 0.7, 1.0, 0.90, 0.01)
        batch_size = st.slider("Batch Size (units)", 50, 500, 200)
        temperature = st.slider("Temperature (°C)", 15.0, 35.0, 22.0, 0.5)
        humidity = st.slider("Humidity (%)", 30.0, 80.0, 50.0, 1.0)

    shift_choice = st.radio("Shift", ["Morning", "Afternoon", "Night"], horizontal=True)
    shift_encoded = {"Morning": 2, "Afternoon": 1, "Night": 0}[shift_choice]

    # Build raw feature dict → engineer
    raw_input = pd.DataFrame([{
        "shift": shift_choice,
        "shift_duration": shift_duration,
        "machine_age": machine_age,
        "equipment_utilization": equipment_utilization,
        "maintenance_frequency": maintenance_frequency,
        "num_workers": num_workers,
        "operator_skill_level": operator_skill_level,
        "raw_material_quality": raw_material_quality,
        "defect_rate": defect_rate,
        "temperature": temperature,
        "humidity": humidity,
        "production_line_speed": production_line_speed,
        "downtime_hours": downtime_hours,
        "batch_size": batch_size,
        "energy_consumption": energy_consumption,
        "quality_check_frequency": quality_check_frequency,
        "rework_rate": rework_rate,
        "material_availability": material_availability,
    }])
    engineered_input = engineer_features(raw_input)
    X_input = engineered_input[feature_names]
    prediction_value = float(pred_model.predict(X_input)[0])

    # All model predictions
    all_preds = {name: float(m.predict(X_input)[0]) for name, m in fitted.items()}

    st.markdown("---")
    st.subheader("🔮 Prediction Results")

    # KPI
    col_a, col_b, col_c = st.columns(3)
    col_a.metric(f"Predicted Throughput ({pred_model_name})", f"{prediction_value:,.0f} units")
    col_b.metric("Dataset Mean Throughput", f"{df['throughput'].mean():,.0f} units")
    delta_pct = (prediction_value - df["throughput"].mean()) / df["throughput"].mean() * 100
    col_c.metric("Δ vs Average", f"{delta_pct:+.1f}%")

    col_l, col_r = st.columns(2)

    # ── VIZ 24: All model predictions bar ──
    with col_l:
        st.subheader("24. All Model Predictions")
        fig = px.bar(
            x=list(all_preds.keys()), y=list(all_preds.values()),
            color=list(all_preds.keys()), color_discrete_map=MODEL_COLORS,
            labels={"x": "Model", "y": "Predicted Throughput"},
            title="Throughput Predicted by All Models",
            text=[f"{v:,.0f}" for v in all_preds.values()],
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    # ── VIZ 25: Prediction in context of distribution ──
    with col_r:
        st.subheader("25. Prediction vs Historical Distribution")
        fig = px.histogram(
            df, x="throughput", nbins=50, color_discrete_sequence=["lightgray"],
            labels={"throughput": "Throughput"}, title="Where Does Your Prediction Land?",
        )
        fig.add_vline(
            x=prediction_value, line_dash="dash", line_color=MODEL_COLORS[pred_model_name],
            annotation_text=f"{pred_model_name}: {prediction_value:,.0f}",
            annotation_position="top right",
        )
        fig.add_vline(
            x=df["throughput"].mean(), line_dash="dot", line_color="black",
            annotation_text="Dataset Mean", annotation_position="top left",
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Efficiency summary ──
    eff_index = (
        equipment_utilization * raw_material_quality * material_availability
        * (1 - defect_rate) * (1 - rework_rate)
    )
    st.markdown("---")
    st.subheader("📋 Efficiency Summary")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Efficiency Index", f"{eff_index:.3f}")
    c2.metric("Net Productive Time", f"{max(shift_duration - downtime_hours, 0):.1f} hrs")
    c3.metric("Workforce Productivity", f"{num_workers * operator_skill_level:.0f}")
    c4.metric("Machine Health Score", f"{min(maintenance_frequency / (machine_age + 1), 10):.2f}")

    # ── VIZ 26: Gauge for efficiency index ──
    st.subheader("26. Efficiency Index Gauge")
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=eff_index,
        number={"valueformat": ".3f"},
        gauge={
            "axis": {"range": [0, 1]},
            "bar": {"color": "steelblue"},
            "steps": [
                {"range": [0, 0.4], "color": "lightcoral"},
                {"range": [0.4, 0.7], "color": "lightyellow"},
                {"range": [0.7, 1.0], "color": "lightgreen"},
            ],
            "threshold": {"line": {"color": "green", "width": 4}, "thickness": 0.75, "value": 0.8},
        },
        title={"text": "Overall Production Efficiency Index"},
    ))
    st.plotly_chart(fig, use_container_width=True)
