"""Synthetic manufacturing data generator for throughput prediction."""

import numpy as np
import pandas as pd


def generate_manufacturing_data(n_samples: int = 2000, random_state: int = 42) -> pd.DataFrame:
    """Generate synthetic manufacturing production data.

    Args:
        n_samples: Number of data samples to generate.
        random_state: Random seed for reproducibility.

    Returns:
        DataFrame with manufacturing features and throughput target.
    """
    rng = np.random.RandomState(random_state)

    # Time / shift features
    shifts = rng.choice(["Morning", "Afternoon", "Night"], size=n_samples, p=[0.4, 0.35, 0.25])
    shift_duration = np.where(
        shifts == "Night",
        rng.uniform(6, 8, n_samples),
        rng.uniform(7, 10, n_samples),
    )

    # Equipment / machine features
    machine_age = rng.uniform(0.5, 15, n_samples)          # years
    equipment_utilization = rng.uniform(0.50, 1.0, n_samples)  # fraction
    maintenance_frequency = rng.randint(1, 13, n_samples)   # times per year

    # Workforce features
    num_workers = rng.randint(5, 31, n_samples)
    operator_skill_level = rng.uniform(1, 10, n_samples)    # 1-10 score

    # Material / process features
    raw_material_quality = rng.uniform(0.5, 1.0, n_samples)  # fraction
    defect_rate = rng.uniform(0.01, 0.15, n_samples)         # fraction

    # Environmental features
    temperature = rng.uniform(15, 35, n_samples)             # Celsius
    humidity = rng.uniform(30, 80, n_samples)                # percent

    # Production line features
    production_line_speed = rng.uniform(50, 150, n_samples)  # units/min
    downtime_hours = rng.uniform(0, 3, n_samples)            # hours per shift
    batch_size = rng.randint(50, 501, n_samples)             # units per batch
    energy_consumption = rng.uniform(100, 500, n_samples)    # kWh per shift

    # Quality control
    quality_check_frequency = rng.randint(1, 11, n_samples)  # checks per shift
    rework_rate = rng.uniform(0.0, 0.10, n_samples)          # fraction

    # Supply chain
    material_availability = rng.uniform(0.7, 1.0, n_samples)  # fraction

    # ---- Throughput formula (units per shift, deterministic + noise) ----
    shift_multiplier = np.where(shifts == "Morning", 1.05,
                       np.where(shifts == "Afternoon", 1.0, 0.90))

    throughput = (
        production_line_speed * shift_duration * equipment_utilization
        + batch_size * material_availability * raw_material_quality
        + num_workers * operator_skill_level * 10
        - machine_age * 15
        + maintenance_frequency * 20
        - defect_rate * 500
        - rework_rate * 400
        - downtime_hours * 100
        - np.abs(temperature - 22) * 5
        - np.abs(humidity - 50) * 2
        + quality_check_frequency * 8
    ) * shift_multiplier

    # Add realistic noise
    noise = rng.normal(0, throughput * 0.05)
    throughput = np.maximum(throughput + noise, 50)  # minimum 50 units

    # Date range for time-series visualizations
    dates = pd.date_range(start="2023-01-01", periods=n_samples, freq="h")

    df = pd.DataFrame(
        {
            "date": dates,
            "shift": shifts,
            "shift_duration": np.round(shift_duration, 2),
            "machine_age": np.round(machine_age, 2),
            "equipment_utilization": np.round(equipment_utilization, 3),
            "maintenance_frequency": maintenance_frequency,
            "num_workers": num_workers,
            "operator_skill_level": np.round(operator_skill_level, 2),
            "raw_material_quality": np.round(raw_material_quality, 3),
            "defect_rate": np.round(defect_rate, 4),
            "temperature": np.round(temperature, 1),
            "humidity": np.round(humidity, 1),
            "production_line_speed": np.round(production_line_speed, 1),
            "downtime_hours": np.round(downtime_hours, 2),
            "batch_size": batch_size,
            "energy_consumption": np.round(energy_consumption, 1),
            "quality_check_frequency": quality_check_frequency,
            "rework_rate": np.round(rework_rate, 4),
            "material_availability": np.round(material_availability, 3),
            "throughput": np.round(throughput, 1),
        }
    )
    return df
