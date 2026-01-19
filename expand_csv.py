# expand_csv.py  -- expand BIOSAFETY_EL.csv to ~130 rows with extra features

import pandas as pd
import numpy as np

# 1. Load your existing synthetic file
src = "BIOSAFETY_EL.csv"          # must be in same folder
df = pd.read_csv(src)

print("Original rows:", len(df))

# 2. Target size
n_target = 130        # choose any number between 100 and 150
n_current = len(df)
extra_needed = max(0, n_target - n_current)

# Columns
cat_cols = [
    "food_type",
    "customers_per_day",
    "water_source",
    "change_of_oil",
    "hand_hygiene",
    "utensil_hygiene",
    "covered_food",
    "proximity_to_waste",
    "surface_moisture_level",
    "storage_type",
]

num_cols = [
    "hours_per_day",
    "ambient_temperature_c",
    "relative_humidity_pct",
    "time_since_preparation_hours",
]

# 3. Generate extra rows by sampling existing ones + small noise
rows = []
rng = np.random.default_rng(seed=42)

for _ in range(extra_needed):
    base = df.sample(1, replace=True).iloc[0].copy()

    # jitter numeric values within realistic Bangalore street ranges
    base["hours_per_day"] = int(np.clip(
        base["hours_per_day"] + rng.integers(-1, 2),
        4, 12
    ))
    base["ambient_temperature_c"] = float(np.clip(
        base["ambient_temperature_c"] + rng.normal(0, 0.8),
        28, 40
    ))
    base["relative_humidity_pct"] = int(np.clip(
        base["relative_humidity_pct"] + rng.normal(0, 3),
        45, 85
    ))
    base["time_since_preparation_hours"] = float(np.clip(
        base["time_since_preparation_hours"] + rng.normal(0, 0.5),
        0.5, 8
    ))

    rows.append(base)

extra_df = pd.DataFrame(rows)
full = pd.concat([df, extra_df], ignore_index=True)

# 4. New spoilage‑related parameters, tied to existing ones
full["stall_temperature_c"] = full["ambient_temperature_c"] + rng.normal(0, 0.7, len(full))

full["food_core_temp_c"] = np.clip(
    full["ambient_temperature_c"] + rng.normal(-2, 1.5, len(full)),
    20, 80
)

# proxy for water activity (aw) based on surface moisture
aw_high = rng.uniform(0.92, 0.99, len(full))
aw_med = rng.uniform(0.85, 0.93, len(full))
aw_low = rng.uniform(0.75, 0.86, len(full))

full["surface_aw_index"] = np.where(
    full["surface_moisture_level"] == "High",
    aw_high,
    np.where(
        full["surface_moisture_level"] == "Medium",
        aw_med,
        aw_low
    ),
)

# distance to drain/waste based on proximity category
dist_high = rng.uniform(0.5, 2.0, len(full))
dist_med = rng.uniform(2.0, 5.0, len(full))
dist_low = rng.uniform(5.0, 15.0, len(full))

full["distance_to_drain_m"] = np.where(
    full["proximity_to_waste"] == "High",
    dist_high,
    np.where(
        full["proximity_to_waste"] == "Medium",
        dist_med,
        dist_low
    ),
)

# flies presence more likely near waste
flies_high = rng.choice(["Yes", "No"], size=len(full), p=[0.8, 0.2])
flies_other = rng.choice(["Yes", "No"], size=len(full), p=[0.3, 0.7])

full["flies_present"] = np.where(
    full["proximity_to_waste"] == "High",
    flies_high,
    flies_other,
)

# cleaning frequency higher when hand_hygiene is High
full["cleaning_frequency_per_day"] = np.where(
    full["hand_hygiene"] == "High",
    rng.integers(4, 8, len(full)),
    rng.integers(1, 5, len(full)),
)

# 5. Rebuild vendor_id sequence
full["vendor_id"] = [f"V{i+1:03d}" for i in range(len(full))]

# 6. Recompute risk_label with same rule as training script
risk_score = (
    (full["ambient_temperature_c"] > 33).astype(int) +
    (full["relative_humidity_pct"] > 65).astype(int) +
    (full["time_since_preparation_hours"] > 3).astype(int) +
    (full["covered_food"].str.lower() == "low").astype(int) +
    (full["proximity_to_waste"].str.lower() == "high").astype(int)
)

def bucket(score: int) -> str:
    if score <= 1:
        return "Low"
    elif score <= 3:
        return "Medium"
    else:
        return "High"

full["risk_label"] = risk_score.apply(bucket)

# 7. Save expanded file
out_path = "BIOSAFETY_EL_expanded.csv"
full.to_csv(out_path, index=False)

print("Expanded rows:", len(full))
print("Saved to", out_path)
print("\nPreview:\n", full.head())
