import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────
# PAGE CONFIG & GLOBAL STYLING
# ──────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🏭 Manufacturing Throughput Predictor",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    /* ── Metric cards ── */
    .metric-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 24px;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1e40af;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #64748b;
        margin-top: 4px;
    }
    
    /* ── Section header ── */
    .section-header {
        font-size: 1.6rem;
        font-weight: 700;
        margin-top: 2rem;
        margin-bottom: 0.5rem;
        padding-bottom: 8px;
        border-bottom: 3px solid #2563eb;
        color: #1e293b;
    }
    
    /* ── Info boxes ── */
    .info-box {
        background: #f0f9ff;
        border-left: 4px solid #2563eb;
        border-radius: 8px;
        padding: 16px 20px;
        margin: 10px 0 20px 0;
        color: #334155;
        line-height: 1.7;
    }
    
    /* ── Hero banner ── */
    .hero-banner {
        background: #1e40af;
        border-radius: 16px;
        padding: 40px;
        text-align: center;
        margin-bottom: 30px;
    }
    .hero-title {
        font-size: 2.8rem;
        font-weight: 900;
        color: #ffffff;
        margin-bottom: 8px;
    }
    .hero-subtitle {
        font-size: 1.1rem;
        color: #dbeafe;
    }
    
    /* ── Step cards ── */
    .step-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 12px;
    }
    .step-card:hover { border-left: 4px solid #2563eb; }
    .step-number {
        display: inline-block;
        background: #2563eb;
        color: white;
        font-weight: 800;
        border-radius: 50%;
        width: 32px;
        height: 32px;
        text-align: center;
        line-height: 32px;
        margin-right: 10px;
    }
    
    /* ── Comparison table ── */
    .comparison-winner {
        background: #f0fdf4;
        border: 2px solid #16a34a;
        border-radius: 12px;
        padding: 16px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────
# DATA GENERATION & CACHING
# ──────────────────────────────────────────────────────────────────
@st.cache_data
def generate_data(records=2000, seed=42):
    np.random.seed(seed)
    machine_speed = np.random.randint(60, 121, records)
    shift_hours = np.random.randint(6, 11, records)
    downtime = np.random.rand(records) * 2
    defect_rate = np.random.rand(records) * 0.1
    operator_efficiency = np.random.rand(records) * 0.3 + 0.7
    maintenance_delay = np.random.rand(records) * 1

    effective_time = shift_hours - downtime
    throughput = (
        machine_speed * effective_time * operator_efficiency
        - (machine_speed * defect_rate)
        - (maintenance_delay * 10)
    )
    data = pd.DataFrame({
        'machine_speed': machine_speed,
        'shift_hours': shift_hours,
        'downtime': downtime,
        'defect_rate': defect_rate,
        'operator_efficiency': operator_efficiency,
        'maintenance_delay': maintenance_delay,
        'throughput': throughput
    })
    # Feature engineering
    data["effective_time"] = data["shift_hours"] - data["downtime"]
    data["downtime_ratio"] = data["downtime"] / data["shift_hours"]
    data["defect_impact"] = data["machine_speed"] * data["defect_rate"]
    return data


@st.cache_resource
def train_all_models(_data):
    X = _data.drop("throughput", axis=1)
    y = _data["throughput"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=10,
                                               min_samples_split=10, min_samples_leaf=5,
                                               random_state=42),
        "Decision Tree": DecisionTreeRegressor(max_depth=4, random_state=42),
        "Linear Regression": LinearRegression(),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, max_depth=5,
                                                        random_state=42),
    }
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        results[name] = {
            "model": model,
            "predictions": preds,
            "mae": mean_absolute_error(y_test, preds),
            "rmse": np.sqrt(mean_squared_error(y_test, preds)),
            "r2": r2_score(y_test, preds),
        }
    return results, X_train, X_test, y_train, y_test, X.columns.tolist()


# ──────────────────────────────────────────────────────────────────
# SIDEBAR NAVIGATION
# ──────────────────────────────────────────────────────────────────
st.sidebar.markdown("""
<div style='text-align:center; padding: 20px 0 10px 0;'>
    <span style='font-size:3rem;'>🏭</span><br>
    <span style='font-size:1.3rem; font-weight:800; color:#1e40af;'>
        Throughput<br>Predictor
    </span>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")

page = st.sidebar.radio(
    "📍 Navigate",
    [
        "🏠 Home",
        "📊 Data Simulation",
        "🔧 Feature Engineering",
        "🤖 Model Training",
        "📈 Evaluation Metrics",
        "📉 Visualizations",
        "⚔️ Model Comparison",
        "🎯 Prediction Simulator",
    ],
    label_visibility="collapsed",
)

st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style='text-align:center; color:#94a3b8; font-size:0.75rem; padding:10px;'>
    Manufacturing ML Platform<br>
    Built with ❤️ using Streamlit<br>
    © 2026 Production Analytics
</div>
""", unsafe_allow_html=True)

# Load data & models
data = generate_data()
results, X_train, X_test, y_train, y_test, feature_names = train_all_models(data)
best = max(results, key=lambda k: results[k]["r2"])


# ══════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════
def metric_card(label, value, icon="📌"):
    st.markdown(f"""
    <div class="metric-card">
        <div style="font-size:1.8rem;">{icon}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-label">{label}</div>
    </div>
    """, unsafe_allow_html=True)


def section_header(text):
    st.markdown(f'<div class="section-header">{text}</div>', unsafe_allow_html=True)


def info_box(text):
    st.markdown(f'<div class="info-box">{text}</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# PAGE: HOME
# ══════════════════════════════════════════════════════════════════
if page == "🏠 Home":
    st.markdown("""
    <div class="hero-banner">
        <div class="hero-title">🏭 Manufacturing Throughput Prediction</div>
        <div class="hero-subtitle">
            Production Analytics &nbsp;•&nbsp; Bottleneck Detection &nbsp;•&nbsp; Smart Forecasting
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Problem Statement
    section_header("📋 Problem Statement")
    info_box("""
        Factories must predict throughput to <b>avoid bottlenecks</b>. 
        Machine Learning can forecast throughput using <b>machine speed</b>, <b>downtime</b>, 
        <b>operator efficiency</b>, and synthetic production data — enabling <b>proactive scheduling</b> 
        and <b>resource optimization</b>.
    """)

    # Key stats
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        metric_card("Data Points", f"{len(data):,}", "📦")
    with col2:
        metric_card("Features", f"{data.shape[1] - 1}", "🔢")
    with col3:
        best_model = max(results, key=lambda k: results[k]["r2"])
        metric_card("Best R² Score", f"{results[best_model]['r2']:.4f}", "🏆")
    with col4:
        metric_card("Models Trained", f"{len(results)}", "🤖")

    st.markdown("<br>", unsafe_allow_html=True)

    # Pipeline overview
    section_header("🔄 ML Pipeline Overview")
    steps = [
        ("1", "Data Simulation", "Generate 2,000 synthetic manufacturing records with domain-based rules"),
        ("2", "Feature Engineering", "Create effective_time, downtime_ratio, and defect_impact features"),
        ("3", "Model Training", "Train Random Forest, Decision Tree, Linear Regression & Gradient Boosting"),
        ("4", "Evaluation", "Compare models using MAE, RMSE, and R² Score metrics"),
        ("5", "Visualization", "Interactive charts for feature importance, predictions & bottleneck analysis"),
        ("6", "Prediction", "Real-time throughput prediction using adjustable factory parameters"),
    ]
    for num, title, desc in steps:
        st.markdown(f"""
        <div class="step-card">
            <span class="step-number">{num}</span>
            <b style="color:#1e293b; font-size:1.05rem;">{title}</b>
            <p style="color:#64748b; margin: 6px 0 0 42px; font-size:0.9rem;">{desc}</p>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# PAGE: DATA SIMULATION
# ══════════════════════════════════════════════════════════════════
elif page == "📊 Data Simulation":
    st.markdown("""
    <div class="hero-banner" style="padding:28px; background:#0d9488;">
        <div style="font-size:2rem; font-weight:800; color:#fff;">
            📊 Module 1: Data Simulation
        </div>
        <div style="color:#f0fdfa; font-size:0.95rem; margin-top:6px;">
            Generating synthetic manufacturing data with domain-based logic
        </div>
    </div>
    """, unsafe_allow_html=True)

    info_box("""
        <b>What?</b> Generate synthetic data mimicking real-world factory production lines.<br>
        <b>Why?</b> Real factory data is confidential. Synthetic data lets us embed realistic 
        manufacturing logic (speed, downtime, efficiency) for reproducible ML experiments.<br>
        <b>How?</b> Domain-based rules determine throughput from machine speed, shift hours, 
        downtime, defect rate, operator efficiency, and maintenance delay.
    """)

    # Dataset preview
    section_header("📋 Dataset Preview")
    col1, col2 = st.columns([3, 1])
    with col1:
        st.dataframe(data.head(20), use_container_width=True, height=400)
    with col2:
        st.markdown("**Dataset Shape**")
        st.info(f"🔢 Rows: {data.shape[0]}")
        st.info(f"📊 Columns: {data.shape[1]}")
        st.markdown("**Throughput Stats**")
        st.success(f"📈 Max: {data['throughput'].max():.1f}")
        st.warning(f"📊 Mean: {data['throughput'].mean():.1f}")
        st.error(f"📉 Min: {data['throughput'].min():.1f}")

    # Statistical summary
    section_header("📊 Statistical Summary")
    st.dataframe(data.describe().T.style.format("{:.3f}").background_gradient(cmap="Blues"),
                 use_container_width=True)

    # Distribution plots
    section_header("📈 Feature Distributions")
    raw_features = ['machine_speed', 'shift_hours', 'downtime', 'defect_rate',
                    'operator_efficiency', 'maintenance_delay', 'throughput']
    selected_feat = st.selectbox("Select feature to visualize:", raw_features)

    col1, col2 = st.columns(2)
    with col1:
        fig = px.histogram(data, x=selected_feat, nbins=40, color_discrete_sequence=["#6366f1"],
                           title=f"Distribution of {selected_feat}",
                           template="plotly_dark")
        fig.update_layout(bargap=0.05)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.box(data, y=selected_feat, color_discrete_sequence=["#8b5cf6"],
                     title=f"Box Plot of {selected_feat}",
                     template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

    # Correlation heatmap
    section_header("🔥 Correlation Heatmap")
    corr = data[raw_features].corr()
    fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                    title="Feature Correlation Matrix", template="plotly_dark",
                    aspect="auto")
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════
# PAGE: FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════
elif page == "🔧 Feature Engineering":
    st.markdown("""
    <div class="hero-banner" style="padding:28px; background:#16a34a;">
        <div style="font-size:2rem; font-weight:800; color:#fff;">
            🔧 Module 2: Feature Engineering
        </div>
        <div style="color:#f0fdf4; font-size:0.95rem; margin-top:6px;">
            Creating meaningful features from raw production data
        </div>
    </div>
    """, unsafe_allow_html=True)

    info_box("""
        <b>What?</b> Create new high-informative features from existing columns.<br>
        <b>Why?</b> Raw data doesn't capture production logic clearly. Engineered features 
        improve model accuracy and interpretability.<br>
        <b>Features Created:</b><br>
        &nbsp;&nbsp;• <b>effective_time</b> = shift_hours − downtime (actual productive hours)<br>
        &nbsp;&nbsp;• <b>downtime_ratio</b> = downtime / shift_hours (relative downtime severity)<br>
        &nbsp;&nbsp;• <b>defect_impact</b> = machine_speed × defect_rate (defect loss at given speed)
    """)

    section_header("🧮 Feature Formulas & Impact")
    eng_features = {
        "effective_time": {"formula": "shift_hours − downtime",
                           "meaning": "Actual productive hours after removing downtime"},
        "downtime_ratio": {"formula": "downtime / shift_hours",
                           "meaning": "Proportion of shift lost to machine downtime"},
        "defect_impact": {"formula": "machine_speed × defect_rate",
                          "meaning": "Volume of defective output at current speed"},
    }

    for feat, info in eng_features.items():
        col1, col2, col3 = st.columns([1, 2, 2])
        with col1:
            st.markdown(f"**`{feat}`**")
        with col2:
            st.code(info["formula"], language="text")
        with col3:
            st.caption(info["meaning"])

    # Before & After comparison
    section_header("📊 Before vs After Engineering")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Original Features (6)**")
        st.dataframe(data[['machine_speed', 'shift_hours', 'downtime',
                           'defect_rate', 'operator_efficiency', 'maintenance_delay']].head(10),
                     use_container_width=True)
    with col2:
        st.markdown("**With Engineered Features (9)**")
        st.dataframe(data.drop("throughput", axis=1).head(10), use_container_width=True)

    # Engineered feature distributions
    section_header("📈 Engineered Feature Distributions")
    for feat in ["effective_time", "downtime_ratio", "defect_impact"]:
        col1, col2 = st.columns(2)
        with col1:
            fig = px.histogram(data, x=feat, nbins=40,
                               color_discrete_sequence=["#10b981"],
                               title=f"Distribution of {feat}",
                               template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig = px.scatter(data, x=feat, y="throughput", opacity=0.4,
                             color_discrete_sequence=["#34d399"],
                             title=f"{feat} vs Throughput",
                             template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)

    # Correlation with target
    section_header("🎯 Correlation with Throughput")
    corr_with_target = data.drop("throughput", axis=1).corrwith(data["throughput"]).sort_values(ascending=False)
    fig = px.bar(x=corr_with_target.values, y=corr_with_target.index,
                 orientation='h', color=corr_with_target.values,
                 color_continuous_scale="Viridis",
                 title="Feature Correlation with Throughput",
                 template="plotly_dark",
                 labels={"x": "Correlation", "y": "Feature"})
    fig.update_layout(height=450)
    st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════
# PAGE: MODEL TRAINING
# ══════════════════════════════════════════════════════════════════
elif page == "🤖 Model Training":
    st.markdown("""
    <div class="hero-banner" style="padding:28px; background:#dc2626;">
        <div style="font-size:2rem; font-weight:800; color:#fff;">
            🤖 Module 3: Regression Modeling
        </div>
        <div style="color:#fef2f2; font-size:0.95rem; margin-top:6px;">
            Training multiple ML models to predict production throughput
        </div>
    </div>
    """, unsafe_allow_html=True)

    info_box("""
        <b>What?</b> Train regression models to predict continuous throughput values.<br>
        <b>Why Regression?</b> Throughput is a numeric quantity (units/shift), making regression the correct approach.<br>
        <b>Models Used:</b><br>
        &nbsp;&nbsp;🌲 <b>Random Forest</b> — Ensemble of decision trees, handles non-linearity<br>
        &nbsp;&nbsp;🌳 <b>Decision Tree</b> — Simple, interpretable tree-based model<br>
        &nbsp;&nbsp;📏 <b>Linear Regression</b> — Baseline linear model<br>
        &nbsp;&nbsp;🚀 <b>Gradient Boosting</b> — Sequential boosting for high accuracy
    """)

    # Training configuration
    section_header("⚙️ Training Configuration")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**Data Split**")
        st.info(f"🔵 Training: {len(X_train)} samples (80%)")
        st.info(f"🟠 Testing: {len(X_test)} samples (20%)")
    with col2:
        st.markdown("**Random Forest Params**")
        st.code("n_estimators = 100\nmax_depth = 10\nmin_samples_split = 10\nmin_samples_leaf = 5")
    with col3:
        st.markdown("**Gradient Boosting Params**")
        st.code("n_estimators = 100\nmax_depth = 5\nlearning_rate = 0.1")

    # Model results overview
    section_header("📊 Model Performance Overview")
    cols = st.columns(4)
    icons = {"Random Forest": "🌲", "Decision Tree": "🌳",
             "Linear Regression": "📏", "Gradient Boosting": "🚀"}
    for i, (name, res) in enumerate(results.items()):
        with cols[i]:
            st.markdown(f"""
            <div class="metric-card">
                <div style="font-size:2rem;">{icons[name]}</div>
                <div style="color:#1e293b; font-weight:700; font-size:1rem; margin:8px 0;">{name}</div>
                <div class="metric-value">{res['r2']:.4f}</div>
                <div class="metric-label">R² Score</div>
                <hr style="border-color:#e2e8f0; margin:10px 0;">
                <div style="color:#64748b; font-size:0.8rem;">
                    MAE: {res['mae']:.2f}<br>RMSE: {res['rmse']:.2f}
                </div>
            </div>
            """, unsafe_allow_html=True)

    # Feature importance for tree models
    section_header("🌲 Random Forest — Feature Importance")
    rf_model = results["Random Forest"]["model"]
    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": rf_model.feature_importances_
    }).sort_values(by="Importance", ascending=True)

    fig = px.bar(importance_df, x="Importance", y="Feature", orientation="h",
                 color="Importance", color_continuous_scale="Plasma",
                 title="Feature Importance (Random Forest)",
                 template="plotly_dark")
    fig.update_layout(height=450, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════
# PAGE: EVALUATION METRICS
# ══════════════════════════════════════════════════════════════════
elif page == "📈 Evaluation Metrics":
    st.markdown("""
    <div class="hero-banner" style="padding:28px; background:#9333ea;">
        <div style="font-size:2rem; font-weight:800; color:#fff;">
            📈 Module 4: Evaluation Metrics
        </div>
        <div style="color:#f5f3ff; font-size:0.95rem; margin-top:6px;">
            Measuring model reliability with industry-standard metrics
        </div>
    </div>
    """, unsafe_allow_html=True)

    info_box("""
        <b>Why Evaluation?</b> Prevents overfitting, validates reliability, and quantifies accuracy.<br><br>
        <b>Metrics Explained:</b><br>
        &nbsp;&nbsp;📏 <b>MAE</b> (Mean Absolute Error) → Average prediction error in throughput units<br>
        &nbsp;&nbsp;📐 <b>RMSE</b> (Root Mean Squared Error) → Penalizes large errors more heavily<br>
        &nbsp;&nbsp;🎯 <b>R² Score</b> → How well the model explains data variance (1.0 = perfect)
    """)

    # Detailed metrics table
    section_header("📊 Detailed Metrics Comparison")
    metrics_df = pd.DataFrame({
        "Model": list(results.keys()),
        "MAE": [r["mae"] for r in results.values()],
        "RMSE": [r["rmse"] for r in results.values()],
        "R² Score": [r["r2"] for r in results.values()],
    }).sort_values("R² Score", ascending=False).reset_index(drop=True)
    metrics_df.index = metrics_df.index + 1
    metrics_df.index.name = "Rank"

    st.dataframe(
        metrics_df.style.format({"MAE": "{:.4f}", "RMSE": "{:.4f}", "R² Score": "{:.6f}"})
        .background_gradient(subset=["R² Score"], cmap="Greens")
        .background_gradient(subset=["MAE", "RMSE"], cmap="Reds_r"),
        use_container_width=True
    )

    # Grouped bar chart
    section_header("📊 Metrics Comparison Charts")
    col1, col2 = st.columns(2)
    with col1:
        fig = go.Figure()
        colors = ["#6366f1", "#10b981", "#f59e0b", "#ef4444"]
        for i, (name, res) in enumerate(results.items()):
            fig.add_trace(go.Bar(name=name, x=["MAE", "RMSE"],
                                 y=[res["mae"], res["rmse"]],
                                 marker_color=colors[i]))
        fig.update_layout(barmode="group", title="MAE & RMSE Comparison",
                          template="plotly_dark", height=400)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        r2_df = pd.DataFrame({
            "Model": list(results.keys()),
            "R² Score": [r["r2"] for r in results.values()]
        })
        fig = px.bar(r2_df, x="Model", y="R² Score", color="Model",
                     color_discrete_sequence=colors,
                     title="R² Score Comparison",
                     template="plotly_dark")
        fig.update_layout(height=400, showlegend=False)
        fig.add_hline(y=1.0, line_dash="dash", line_color="#4ade80",
                      annotation_text="Perfect Score (1.0)")
        st.plotly_chart(fig, use_container_width=True)

    # Error distribution
    section_header("📉 Prediction Error Distribution")
    selected_model = st.selectbox("Select model:", list(results.keys()))
    errors = y_test.values - results[selected_model]["predictions"]

    col1, col2 = st.columns(2)
    with col1:
        fig = px.histogram(x=errors, nbins=50, color_discrete_sequence=["#8b5cf6"],
                           title=f"Error Distribution — {selected_model}",
                           labels={"x": "Prediction Error"},
                           template="plotly_dark")
        fig.add_vline(x=0, line_dash="dash", line_color="#ef4444")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.box(x=errors, color_discrete_sequence=["#6366f1"],
                     title=f"Error Box Plot — {selected_model}",
                     labels={"x": "Prediction Error"},
                     template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════
# PAGE: VISUALIZATIONS
# ══════════════════════════════════════════════════════════════════
elif page == "📉 Visualizations":
    st.markdown("""
    <div class="hero-banner" style="padding:28px; background:#2563eb;">
        <div style="font-size:2rem; font-weight:800; color:#fff;">
            📉 Module 5: Interactive Visualizations
        </div>
        <div style="color:#eff6ff; font-size:0.95rem; margin-top:6px;">
            Deep visual insights into production factors and predictions
        </div>
    </div>
    """, unsafe_allow_html=True)

    info_box("""
        <b>Purpose:</b> Visual representation of model results for better understanding.<br>
        <b>Charts:</b> Feature Importance, Actual vs Predicted, Residual Plots, 3D scatter, and more.
    """)

    vis_model = st.selectbox("Select Model for Visualization:", list(results.keys()))
    preds = results[vis_model]["predictions"]

    # Actual vs Predicted
    section_header("🎯 Actual vs Predicted Throughput")
    col1, col2 = st.columns(2)
    with col1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=y_test.values, y=preds, mode="markers",
                                 marker=dict(color=preds, colorscale="Viridis",
                                             size=5, opacity=0.6, colorbar=dict(title="Predicted")),
                                 name="Predictions"))
        min_val = min(y_test.min(), preds.min())
        max_val = max(y_test.max(), preds.max())
        fig.add_trace(go.Scatter(x=[min_val, max_val], y=[min_val, max_val],
                                 mode="lines", line=dict(color="#ef4444", dash="dash"),
                                 name="Perfect Prediction"))
        fig.update_layout(title=f"Actual vs Predicted — {vis_model}",
                          xaxis_title="Actual", yaxis_title="Predicted",
                          template="plotly_dark", height=480)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Residual plot
        residuals = y_test.values - preds
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=preds, y=residuals, mode="markers",
                                 marker=dict(color=residuals, colorscale="RdBu",
                                             size=5, opacity=0.6),
                                 name="Residuals"))
        fig.add_hline(y=0, line_dash="dash", line_color="#4ade80")
        fig.update_layout(title=f"Residual Plot — {vis_model}",
                          xaxis_title="Predicted", yaxis_title="Residual",
                          template="plotly_dark", height=480)
        st.plotly_chart(fig, use_container_width=True)

    # 3D scatter
    section_header("🌐 3D Production Analysis")
    col1, col2, col3 = st.columns(3)
    all_features = feature_names
    with col1:
        x_feat = st.selectbox("X Axis:", all_features, index=all_features.index("machine_speed"))
    with col2:
        y_feat = st.selectbox("Y Axis:", all_features, index=all_features.index("effective_time"))
    with col3:
        z_feat = st.selectbox("Z Axis:", ["throughput"] + all_features, index=0)

    fig = px.scatter_3d(data, x=x_feat, y=y_feat, z=z_feat,
                        color="throughput", color_continuous_scale="Turbo",
                        opacity=0.5, title=f"3D: {x_feat} × {y_feat} × {z_feat}",
                        template="plotly_dark")
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)

    # Pair plot – throughput vs all features
    section_header("📊 Throughput vs Key Features")
    key_feats = ["machine_speed", "effective_time", "operator_efficiency", "downtime"]
    cols = st.columns(2)
    for i, feat in enumerate(key_feats):
        with cols[i % 2]:
            fig = px.scatter(data, x=feat, y="throughput", opacity=0.3,
                             color="throughput", color_continuous_scale="Viridis",
                             trendline="ols",
                             title=f"Throughput vs {feat}",
                             template="plotly_dark")
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════
# PAGE: MODEL COMPARISON
# ══════════════════════════════════════════════════════════════════
elif page == "⚔️ Model Comparison":
    st.markdown("""
    <div class="hero-banner" style="padding:28px; background:#0369a1;">
        <div style="font-size:2rem; font-weight:800; color:#fff;">
            ⚔️ Head-to-Head Model Comparison
        </div>
        <div style="color:#f0f9ff; font-size:0.95rem; margin-top:6px;">
            Compare all trained models side-by-side
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Winner announcement
    best = max(results, key=lambda k: results[k]["r2"])
    st.markdown(f"""
    <div class="comparison-winner">
        <div style="font-size:2.5rem;">🏆</div>
        <div style="font-size:1.5rem; font-weight:800; color:#16a34a; margin:6px 0;">
            {best}
        </div>
        <div style="color:#374151;">
            Best Model — R² = {results[best]['r2']:.6f} | MAE = {results[best]['mae']:.4f} | RMSE = {results[best]['rmse']:.4f}
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Radar chart
    section_header("🕸️ Performance Radar")
    # Normalize metrics for radar (inverted for error metrics)
    max_mae = max(r["mae"] for r in results.values())
    max_rmse = max(r["rmse"] for r in results.values())
    categories = ["R² Score", "1 − MAE (norm)", "1 − RMSE (norm)", "Accuracy", "R² Score"]

    fig = go.Figure()
    colors = ["#6366f1", "#10b981", "#f59e0b", "#ef4444"]
    for i, (name, res) in enumerate(results.items()):
        vals = [
            res["r2"],
            1 - res["mae"] / max_mae,
            1 - res["rmse"] / max_rmse,
            res["r2"],
            res["r2"],
        ]
        fig.add_trace(go.Scatterpolar(r=vals, theta=categories,
                                       fill="toself", name=name,
                                       line_color=colors[i], opacity=0.7))
    fig.update_layout(polar=dict(bgcolor="#f8fafc",
                                  radialaxis=dict(visible=True, range=[0, 1])),
                      template="plotly_white", height=500,
                      title="Model Performance Radar Chart")
    st.plotly_chart(fig, use_container_width=True)

    # Side-by-side actual vs predicted
    section_header("🎯 Actual vs Predicted — All Models")
    fig = make_subplots(rows=2, cols=2,
                        subplot_titles=list(results.keys()),
                        vertical_spacing=0.12, horizontal_spacing=0.08)

    for i, (name, res) in enumerate(results.items()):
        row = i // 2 + 1
        col = i % 2 + 1
        fig.add_trace(go.Scatter(x=y_test.values, y=res["predictions"],
                                 mode="markers", marker=dict(size=3, opacity=0.5,
                                                              color=colors[i]),
                                 name=name, showlegend=True), row=row, col=col)
        min_v = min(y_test.min(), res["predictions"].min())
        max_v = max(y_test.max(), res["predictions"].max())
        fig.add_trace(go.Scatter(x=[min_v, max_v], y=[min_v, max_v],
                                 mode="lines", line=dict(color="#ef4444", dash="dash"),
                                 showlegend=False), row=row, col=col)

    fig.update_layout(height=700, template="plotly_dark",
                      title="Actual vs Predicted — All Models")
    st.plotly_chart(fig, use_container_width=True)

    # Feature importance comparison (tree-based models)
    section_header("🌲 Feature Importance — Tree-Based Models")
    tree_models = {k: v for k, v in results.items() if k != "Linear Regression"}
    fig = go.Figure()
    for i, (name, res) in enumerate(tree_models.items()):
        imp = res["model"].feature_importances_
        fig.add_trace(go.Bar(name=name, x=feature_names, y=imp,
                             marker_color=colors[i]))
    fig.update_layout(barmode="group", template="plotly_dark", height=450,
                      title="Feature Importance Comparison",
                      xaxis_title="Feature", yaxis_title="Importance")
    st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════
# PAGE: PREDICTION SIMULATOR
# ══════════════════════════════════════════════════════════════════
elif page == "🎯 Prediction Simulator":
    st.markdown("""
    <div class="hero-banner" style="padding:28px; background:#059669;">
        <div style="font-size:2rem; font-weight:800; color:#fff;">
            🎯 Real-Time Throughput Prediction Simulator
        </div>
        <div style="color:#f0fdf4; font-size:0.95rem; margin-top:6px;">
            Adjust factory parameters and get instant ML predictions
        </div>
    </div>
    """, unsafe_allow_html=True)

    info_box("""
        <b>How to use:</b> Adjust the sliders below to simulate different factory conditions. 
        The trained ML models will instantly predict the expected throughput. 
        This helps factory managers <b>plan production</b>, <b>identify bottlenecks</b>, 
        and <b>optimize operations</b> before making real changes.
    """)

    section_header("🎛️ Factory Parameters")
    col1, col2, col3 = st.columns(3)
    with col1:
        machine_speed = st.slider("⚡ Machine Speed (units/hr)", 60, 120, 90, 1)
        shift_hours = st.slider("⏱️ Shift Hours", 6, 10, 8, 1)
    with col2:
        downtime = st.slider("🔴 Downtime (hours)", 0.0, 2.0, 0.5, 0.1)
        defect_rate = st.slider("❌ Defect Rate", 0.00, 0.10, 0.03, 0.01)
    with col3:
        operator_eff = st.slider("👷 Operator Efficiency", 0.70, 1.00, 0.85, 0.01)
        maint_delay = st.slider("🔧 Maintenance Delay (hrs)", 0.0, 1.0, 0.3, 0.1)

    # Compute engineered features
    eff_time = shift_hours - downtime
    dt_ratio = downtime / shift_hours
    def_impact = machine_speed * defect_rate

    input_df = pd.DataFrame([{
        "machine_speed": machine_speed,
        "shift_hours": shift_hours,
        "downtime": downtime,
        "defect_rate": defect_rate,
        "operator_efficiency": operator_eff,
        "maintenance_delay": maint_delay,
        "effective_time": eff_time,
        "downtime_ratio": dt_ratio,
        "defect_impact": def_impact,
    }])

    st.markdown("<br>", unsafe_allow_html=True)
    section_header("🔮 Predictions")

    cols = st.columns(4)
    icons_pred = {"Random Forest": "🌲", "Decision Tree": "🌳",
                  "Linear Regression": "📏", "Gradient Boosting": "🚀"}
    predictions_dict = {}
    for i, (name, res) in enumerate(results.items()):
        pred = res["model"].predict(input_df)[0]
        predictions_dict[name] = pred
        with cols[i]:
            metric_card(name, f"{pred:.1f}", icons_pred[name])

    st.markdown("<br>", unsafe_allow_html=True)

    # Gauge chart for best model
    section_header("📊 Throughput Gauge (Best Model)")
    best_pred = predictions_dict[best]
    max_throughput = data["throughput"].max()

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=best_pred,
        title={"text": f"Predicted Throughput ({best})", "font": {"size": 18, "color": "#e2e8f0"}},
        delta={"reference": data["throughput"].mean(), "suffix": " vs avg"},
        gauge={
            "axis": {"range": [data["throughput"].min(), max_throughput],
                     "tickcolor": "#64748b"},
            "bar": {"color": "#2563eb"},
            "bgcolor": "#f1f5f9",
            "bordercolor": "#e2e8f0",
            "steps": [
                {"range": [data["throughput"].min(), data["throughput"].quantile(0.33)],
                 "color": "#fecaca"},
                {"range": [data["throughput"].quantile(0.33), data["throughput"].quantile(0.66)],
                 "color": "#fef08a"},
                {"range": [data["throughput"].quantile(0.66), max_throughput],
                 "color": "#bbf7d0"},
            ],
            "threshold": {
                "line": {"color": "#ef4444", "width": 3},
                "thickness": 0.8,
                "value": data["throughput"].mean(),
            },
        },
    ))
    fig.update_layout(height=350, template="plotly_white",
                      paper_bgcolor="#ffffff", font_color="#1e293b")
    st.plotly_chart(fig, use_container_width=True)

    # Input summary
    section_header("📋 Input Summary")
    col1, col2 = st.columns(2)
    with col1:
        st.dataframe(input_df.T.rename(columns={0: "Value"}).style.format("{:.4f}"),
                     use_container_width=True)
    with col2:
        # Comparison bar
        fig = px.bar(x=list(predictions_dict.keys()), y=list(predictions_dict.values()),
                     color=list(predictions_dict.keys()),
                     color_discrete_sequence=["#6366f1", "#10b981", "#f59e0b", "#ef4444"],
                     title="Model Predictions Comparison",
                     labels={"x": "Model", "y": "Predicted Throughput"},
                     template="plotly_dark")
        fig.update_layout(height=350, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
