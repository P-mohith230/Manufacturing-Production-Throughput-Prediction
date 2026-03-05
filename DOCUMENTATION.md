# 🏭 Manufacturing Throughput Prediction — Streamlit Platform Documentation

---

## 📋 Table of Contents

1. [Project Overview](#project-overview)
2. [Problem Statement](#problem-statement)
3. [Platform Architecture](#platform-architecture)
4. [Pages & Components Breakdown](#pages--components-breakdown)
5. [Visualizations Catalog](#visualizations-catalog)
6. [Tech Stack](#tech-stack)
7. [How to Run](#how-to-run)
8. [File Structure](#file-structure)

---

## 🎯 Project Overview

This is an **AI-powered Manufacturing Analytics Platform** built with **Streamlit** that predicts factory production throughput using Machine Learning. The platform takes synthetic manufacturing data (simulating real factory conditions) and trains multiple regression models to forecast throughput based on production parameters like machine speed, downtime, operator efficiency, and more.

**Key Capabilities:**
- Synthetic data generation with domain-based manufacturing logic
- Feature engineering for enhanced model performance
- Multi-model training & comparison (4 regression models)
- Interactive visualizations (15+ charts)
- Real-time prediction simulator with adjustable factory parameters

---

## 🏗️ Problem Statement

> **Factories must predict throughput to avoid bottlenecks.** Machine Learning can forecast throughput using machine speed, downtime, operator efficiency, and synthetic production data — enabling proactive scheduling and resource optimization.

**Target Variable:** `throughput` (continuous numeric — units produced per shift)

**Input Features (6 raw + 3 engineered = 9 total):**

| Feature | Type | Range | Description |
|---------|------|-------|-------------|
| `machine_speed` | Raw | 60–120 | Machine operating speed (units/hr) |
| `shift_hours` | Raw | 6–10 | Duration of work shift |
| `downtime` | Raw | 0–2 | Hours of machine downtime |
| `defect_rate` | Raw | 0–0.10 | Fraction of defective output |
| `operator_efficiency` | Raw | 0.70–1.00 | Worker performance factor |
| `maintenance_delay` | Raw | 0–1 | Hours lost to maintenance |
| `effective_time` | Engineered | varies | = shift_hours − downtime |
| `downtime_ratio` | Engineered | 0–0.33 | = downtime / shift_hours |
| `defect_impact` | Engineered | varies | = machine_speed × defect_rate |

---

## 🏛️ Platform Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   STREAMLIT FRONTEND                     │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐   │
│  │  Home    │ │  Data    │ │ Feature  │ │  Model   │   │
│  │  Page    │ │Simulation│ │Engineering│ │ Training │   │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘   │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐   │
│  │Evaluation│ │  Visual- │ │  Model   │ │Prediction│   │
│  │ Metrics  │ │ izations │ │Comparison│ │Simulator │   │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘   │
├─────────────────────────────────────────────────────────┤
│                    ML ENGINE (Cached)                     │
│  ┌──────────────┐  ┌──────────────────────────────────┐  │
│  │ Data Gen     │  │ Trained Models (4):              │  │
│  │ (2000 rows)  │  │ RandomForest, DecisionTree,     │  │
│  │ + Feature    │  │ LinearRegression, GradientBoost  │  │
│  │   Engineering│  │                                  │  │
│  └──────────────┘  └──────────────────────────────────┘  │
├─────────────────────────────────────────────────────────┤
│   NumPy │ Pandas │ Scikit-learn │ Plotly │ Streamlit    │
└─────────────────────────────────────────────────────────┘
```

---

## 📄 Pages & Components Breakdown

### Page 1: 🏠 Home

| Component | Type | What It Does |
|-----------|------|-------------|
| Hero Banner | HTML/CSS | Gradient banner with project title and tagline |
| Problem Statement | Info Box | Explains the manufacturing prediction problem |
| KPI Metric Cards (×4) | Custom Cards | Shows Data Points, Features, Best R², Models Trained |
| ML Pipeline Overview | Step Cards (×6) | Animated step-by-step pipeline from data → prediction |

**Purpose:** Landing page that gives a complete overview of the project, problem, and ML pipeline at a glance.

---

### Page 2: 📊 Data Simulation (Module 1)

| Component | Type | What It Does |
|-----------|------|-------------|
| Module Banner | HTML/CSS | Teal gradient header with module description |
| Explanation Box | Info Box | Explains What/Why/How of data simulation |
| Dataset Preview | Dataframe (20 rows) | Interactive scrollable table of synthetic data |
| Dataset Stats | Info Cards | Row count, column count, throughput min/mean/max |
| Statistical Summary | Styled Dataframe | `describe().T` with blue gradient background |
| Feature Selector | Selectbox | Dropdown to pick any feature for visualization |
| Histogram | Plotly Histogram | Distribution of selected feature (40 bins) |
| Box Plot | Plotly Box | Outlier detection for selected feature |
| Correlation Heatmap | Plotly Imshow | 7×7 matrix with RdBu color scale + annotations |

**Purpose:** Explore and understand the raw synthetic manufacturing dataset.

---

### Page 3: 🔧 Feature Engineering (Module 2)

| Component | Type | What It Does |
|-----------|------|-------------|
| Module Banner | HTML/CSS | Green gradient header |
| Feature Formulas Table | 3-column layout | Shows formula + meaning for each engineered feature |
| Before vs After Tables | Side-by-side Dataframes | Original 6 columns vs enhanced 9 columns |
| Engineered Feature Histograms (×3) | Plotly Histograms | Distribution of each new feature |
| Feature vs Throughput Scatter (×3) | Plotly Scatter | Relationship between each new feature and target |
| Correlation Bar Chart | Plotly Horizontal Bar | All features ranked by correlation with throughput |

**Purpose:** Show how raw features are transformed into meaningful production metrics.

---

### Page 4: 🤖 Model Training (Module 3)

| Component | Type | What It Does |
|-----------|------|-------------|
| Module Banner | HTML/CSS | Red-orange gradient header |
| Explanation Box | Info Box | Why regression, why these models |
| Training Config | 3-column layout | Data split info + hyperparameters for RF & GB |
| Model Performance Cards (×4) | Custom Metric Cards | R², MAE, RMSE for each model with icons |
| Feature Importance Bar | Plotly Horizontal Bar | Plasma-colored importance from Random Forest |

**Purpose:** Show how models are configured, trained, and what features matter most.

---

### Page 5: 📈 Evaluation Metrics (Module 4)

| Component | Type | What It Does |
|-----------|------|-------------|
| Module Banner | HTML/CSS | Pink-to-purple gradient |
| Metrics Explanation | Info Box | Explains MAE, RMSE, R² in plain terms |
| Ranked Metrics Table | Styled Dataframe | All models ranked, with green/red gradient coloring |
| Grouped Bar Chart | Plotly Grouped Bar | MAE & RMSE side-by-side for all models |
| R² Score Bar Chart | Plotly Bar | R² with perfect score reference line |
| Model Selector | Selectbox | Pick model for error analysis |
| Error Histogram | Plotly Histogram | Distribution of prediction errors |
| Error Box Plot | Plotly Box | Error spread and outlier detection |

**Purpose:** Quantify and compare model accuracy with multiple evaluation metrics.

---

### Page 6: 📉 Visualizations (Module 5)

| Component | Type | What It Does |
|-----------|------|-------------|
| Module Banner | HTML/CSS | Purple-to-blue gradient |
| Model Selector | Selectbox | Choose model for all charts on this page |
| Actual vs Predicted Scatter | Plotly Scatter | Color-coded predictions with perfect-prediction line |
| Residual Plot | Plotly Scatter | Residuals vs predicted with zero-line reference |
| 3D Scatter (configurable) | Plotly 3D Scatter | Select any 3 features for interactive 3D analysis |
| Feature vs Throughput (×4) | Plotly Scatter + OLS trendline | Key features with regression trendlines |

**Purpose:** Deep visual exploration of model predictions and production factor relationships.

---

### Page 7: ⚔️ Model Comparison

| Component | Type | What It Does |
|-----------|------|-------------|
| Module Banner | HTML/CSS | Blue-to-purple gradient |
| Winner Announcement | Highlighted Card | Best model with R², MAE, RMSE in green box |
| Radar Chart | Plotly Scatterpolar | 4-model overlay radar comparing normalized metrics |
| 4-Panel Actual vs Predicted | Plotly Subplots (2×2) | Side-by-side predictions for all models |
| Feature Importance Comparison | Plotly Grouped Bar | Tree-based models' feature importance overlaid |

**Purpose:** Head-to-head comparison to determine the best model for production use.

---

### Page 8: 🎯 Prediction Simulator

| Component | Type | What It Does |
|-----------|------|-------------|
| Module Banner | HTML/CSS | Teal-to-blue gradient |
| Usage Guide | Info Box | How to use the simulator |
| Parameter Sliders (×6) | Streamlit Sliders | Adjustable: speed, shift, downtime, defect, efficiency, maintenance |
| Prediction Cards (×4) | Custom Metric Cards | Instant predictions from all 4 models |
| Throughput Gauge | Plotly Indicator (Gauge) | Speedometer-style gauge with red/yellow/green zones |
| Input Summary Table | Styled Dataframe | All input parameters + engineered features |
| Predictions Bar Chart | Plotly Bar | All model predictions side-by-side |

**Purpose:** Interactive tool for factory managers to simulate "what-if" scenarios and get instant ML predictions.

---

## 📊 Visualizations Catalog

| # | Visualization | Chart Type | Library | Page |
|---|--------------|-----------|---------|------|
| 1 | Feature Distribution | Histogram | Plotly | Data Simulation |
| 2 | Feature Box Plot | Box Plot | Plotly | Data Simulation |
| 3 | Correlation Heatmap | Heatmap (imshow) | Plotly | Data Simulation |
| 4 | Engineered Feature Histograms (×3) | Histogram | Plotly | Feature Engineering |
| 5 | Feature vs Throughput Scatter (×3) | Scatter | Plotly | Feature Engineering |
| 6 | Correlation with Target Bar | Horizontal Bar | Plotly | Feature Engineering |
| 7 | Feature Importance (RF) | Horizontal Bar | Plotly | Model Training |
| 8 | MAE/RMSE Grouped Bar | Grouped Bar | Plotly | Evaluation Metrics |
| 9 | R² Score Comparison Bar | Bar | Plotly | Evaluation Metrics |
| 10 | Error Distribution Histogram | Histogram | Plotly | Evaluation Metrics |
| 11 | Error Box Plot | Box Plot | Plotly | Evaluation Metrics |
| 12 | Actual vs Predicted Scatter | Scatter | Plotly | Visualizations |
| 13 | Residual Plot | Scatter | Plotly | Visualizations |
| 14 | 3D Production Analysis | 3D Scatter | Plotly | Visualizations |
| 15 | Throughput vs Key Features (×4) | Scatter + OLS | Plotly | Visualizations |
| 16 | Performance Radar Chart | Scatterpolar | Plotly | Model Comparison |
| 17 | 4-Panel Actual vs Predicted | Subplots (2×2) | Plotly | Model Comparison |
| 18 | Feature Importance Comparison | Grouped Bar | Plotly | Model Comparison |
| 19 | Throughput Gauge | Indicator (Gauge) | Plotly | Prediction Simulator |
| 20 | Model Predictions Bar | Bar | Plotly | Prediction Simulator |

**Total: 20+ interactive visualizations** across 8 pages.

---

## 🛠️ Tech Stack

| Technology | Purpose |
|-----------|---------|
| **Python 3.13** | Core programming language |
| **Streamlit** | Web application framework |
| **Plotly** | Interactive visualizations (20+ charts) |
| **Scikit-learn** | ML models (RF, DT, LR, GB) + metrics |
| **Pandas** | Data manipulation & feature engineering |
| **NumPy** | Numerical operations & data simulation |
| **Custom CSS** | Glassmorphism cards, gradients, animations |

---

## 🚀 How to Run

```bash
# 1. Navigate to the project directory
cd "m:\25(A)"

# 2. Activate virtual environment
.venv\Scripts\activate

# 3. Install dependencies (if not already installed)
pip install streamlit plotly scikit-learn pandas numpy

# 4. Run the application
streamlit run app.py

# 5. Open in browser
# → http://localhost:8501
```

---

## 📁 File Structure

```
m:\25(A)\
├── app.py                                  ← Streamlit application (main file)
├── 25(A) (2).ipynb                         ← Original ML notebook
├── manufacturing_throughput_dataset.csv     ← Generated dataset (from notebook)
├── DOCUMENTATION.md                        ← This documentation file
└── .venv\                                  ← Python virtual environment
```

---

## 🔑 Key Design Decisions

1. **`@st.cache_data` for data generation** — Data is generated once and cached, preventing regeneration on every page switch.
2. **`@st.cache_resource` for model training** — All 4 models are trained once and cached as resources, making page navigation instant.
3. **Plotly over Matplotlib** — All charts are interactive (hover, zoom, pan) instead of static images.
4. **Custom CSS theming** — Dark theme with gradient cards, hover animations, and glassmorphism effects for a professional look.
5. **Sidebar navigation** — Radio buttons for clean single-page-app navigation across 8 pages.
6. **Added Gradient Boosting** — Extended beyond the notebook's 3 models by adding GradientBoostingRegressor for better comparison.

---

*Built with ❤️ for Manufacturing ML Analytics*
