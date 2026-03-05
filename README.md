# Manufacturing Production Throughput Prediction

An end-to-end machine learning project that predicts factory production throughput using multiple regression models, served through a fully interactive Streamlit dashboard with 26 Plotly visualizations, feature engineering, model comparison, and a real-time prediction simulator.

---

## Features

- **Four ML models**: Linear Regression, Decision Tree, Random Forest, Gradient Boosting
- **Feature engineering**: 8 derived features (efficiency index, machine health, net productive time, etc.)
- **26 interactive Plotly visualizations** across 5 dashboard pages
- **Real-time prediction simulator** with adjustable sliders for every production parameter
- **Model comparison**: R², RMSE, MAE, MAPE, 5-fold cross-validation, radar chart
- **Feature importance**: side-by-side RF vs Gradient Boosting comparison
- **Learning curves** for each model

---

## Project Structure

```
├── app.py                    # Streamlit dashboard
├── requirements.txt          # Python dependencies
├── src/
│   ├── data_generator.py     # Synthetic manufacturing data generator
│   ├── feature_engineering.py# Feature engineering & data preparation
│   └── models.py             # Model training, evaluation, utilities
└── tests/
    └── test_models.py        # 28 unit tests for all modules
```

---

## Getting Started

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Streamlit dashboard

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`.

---

## Dashboard Pages

| Page | Description |
|------|-------------|
| 📊 Data Overview | Dataset statistics, throughput distribution, time series, correlation heatmap |
| 🔍 Feature Analysis | Scatter matrix, parallel coordinates, 3D scatter, violin plots, density heatmap |
| 🤖 Model Performance | Test-set metrics, actual vs predicted, residuals, cross-validation, radar chart |
| 📈 Model Insights | Feature importances (RF & GB), learning curve, prediction distributions, MAPE gauge |
| 🎯 Real-Time Predictor | Interactive sliders for all 18 input features → instant throughput prediction |

---

## Running Tests

```bash
pip install pytest
pytest tests/ -v
```

---

## Dataset

The synthetic dataset (2,000 records by default, configurable via sidebar) contains 18 production features:

| Category | Features |
|----------|---------|
| Equipment | machine_age, equipment_utilization, maintenance_frequency, production_line_speed, energy_consumption |
| Workforce | num_workers, operator_skill_level |
| Process | shift_duration, downtime_hours, batch_size, quality_check_frequency |
| Material | raw_material_quality, defect_rate, rework_rate, material_availability |
| Environment | temperature, humidity |
| Temporal | shift (Morning / Afternoon / Night) |

**Target**: `throughput` — total units produced per shift.
