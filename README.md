# 🏭 Manufacturing Throughput Prediction

**Predict factory production throughput using Machine Learning to prevent bottlenecks and optimize operations.**

An end-to-end ML project that simulates real-world manufacturing data, engineers production features, trains & compares multiple regression models, and provides an interactive Streamlit dashboard for real-time throughput prediction.

---

## 📌 Problem Statement

Factories need to **predict production throughput** in advance to avoid bottlenecks, plan schedules, and allocate resources efficiently. This project uses Machine Learning to forecast throughput based on factory parameters — machine speed, shift duration, downtime, defect rate, operator efficiency, and maintenance delays.

---

## 🎯 Key Features

- **Synthetic Data Generation** — 2,000 records with domain-based manufacturing logic
- **Feature Engineering** — 3 engineered features: `effective_time`, `downtime_ratio`, `defect_impact`
- **4 ML Models** — Random Forest, Decision Tree, Linear Regression, Gradient Boosting
- **Model Comparison** — Side-by-side evaluation with MAE, RMSE, and R² Score
- **Interactive Dashboard** — 8-page Streamlit app with 20+ Plotly visualizations
- **Prediction Simulator** — Adjust factory parameters via sliders and get instant predictions

---

## 📊 Dataset Features

| Feature | Type | Range | Description |
|---------|------|-------|-------------|
| `machine_speed` | Raw | 60 – 120 | Machine operating speed (units/hr) |
| `shift_hours` | Raw | 6 – 10 | Duration of work shift (hours) |
| `downtime` | Raw | 0 – 2 | Machine downtime (hours) |
| `defect_rate` | Raw | 0 – 0.10 | Fraction of defective output |
| `operator_efficiency` | Raw | 0.70 – 1.00 | Worker performance factor |
| `maintenance_delay` | Raw | 0 – 1 | Hours lost to maintenance |
| `effective_time` | Engineered | — | shift_hours − downtime |
| `downtime_ratio` | Engineered | — | downtime / shift_hours |
| `defect_impact` | Engineered | — | machine_speed × defect_rate |
| **`throughput`** | **Target** | — | **Units produced per shift** |

---

## 🤖 Models & Results

| Model | MAE | RMSE | R² Score |
|-------|-----|------|----------|
| **Random Forest** | Best | Best | **~0.99** |
| Gradient Boosting | Good | Good | ~0.98 |
| Decision Tree | Moderate | Moderate | ~0.95 |
| Linear Regression | Baseline | Baseline | ~0.93 |

> Random Forest achieves the highest R² score, capturing non-linear interactions between production factors.

---

## 🖥️ Streamlit Dashboard

The interactive dashboard has **8 pages**:

| Page | Description |
|------|-------------|
| 🏠 **Home** | Project overview, KPI cards, ML pipeline steps |
| 📊 **Data Simulation** | Dataset preview, distributions, correlation heatmap |
| 🔧 **Feature Engineering** | Formulas, before/after comparison, feature correlations |
| 🤖 **Model Training** | Hyperparameters, performance cards, feature importance |
| 📈 **Evaluation Metrics** | Ranked metrics table, error distribution analysis |
| 📉 **Visualizations** | Actual vs Predicted, residual plots, 3D scatter analysis |
| ⚔️ **Model Comparison** | Radar chart, 4-panel comparison, importance overlay |
| 🎯 **Prediction Simulator** | Adjustable sliders, real-time predictions, throughput gauge |

---

## 🛠️ Tech Stack

| Technology | Role |
|-----------|------|
| Python | Core language |
| Streamlit | Web dashboard framework |
| Plotly | Interactive visualizations |
| Scikit-learn | ML models & evaluation |
| Pandas / NumPy | Data manipulation |
| Matplotlib / Seaborn | Notebook visualizations |

---

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/<your-username>/Manufacturing-Throughput-Prediction.git
cd Manufacturing-Throughput-Prediction

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate       # Windows
# source .venv/bin/activate  # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit dashboard
streamlit run app.py
```

Then open **http://localhost:8501** in your browser.

---

## 📁 Project Structure

```
├── Manufacturing_Throughput_Prediction.ipynb   # ML notebook (data → model → evaluation)
├── app.py                                      # Streamlit dashboard (8 pages, 20+ charts)
├── requirements.txt                            # Python dependencies
├── DOCUMENTATION.md                            # Detailed platform documentation
├── .gitignore                                  # Git ignore rules
└── README.md                                   # This file
```

---

## 📝 ML Pipeline

```
Data Simulation → Feature Engineering → Model Training → Evaluation → Visualization → Prediction
```

1. **Data Simulation** — Generate synthetic manufacturing data using domain-based rules
2. **Feature Engineering** — Create `effective_time`, `downtime_ratio`, `defect_impact`
3. **Model Training** — Train 4 regression models (RF, DT, LR, GB)
4. **Evaluation** — Compare using MAE, RMSE, R² Score
5. **Visualization** — 20+ interactive Plotly charts
6. **Prediction** — Real-time simulator with adjustable factory parameters

---

## 📜 License

This project is for educational and portfolio purposes.

---

*Built for manufacturing analytics and ML showcase.*
