# ML.py  -- FoodGuardML training script using BIOSAFETY_EL_expanded.csv

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# 1. Load CSV
csv_path = "BIOSAFETY_EL_expanded.csv"   # expanded file (130 rows, 22 cols)
df = pd.read_csv(csv_path)

print("Columns in dataset:\n", df.columns.tolist())
print("\nFirst 5 rows:\n", df.head())

print("\nClass distribution:\n", df["risk_label"].value_counts())

# 2. Feature / target split
# Categorical features (string/ordinal)
cat_cols = [
    "food_type",
    "water_source",
    "change_of_oil",
    "hand_hygiene",
    "utensil_hygiene",
    "covered_food",
    "proximity_to_waste",
    "surface_moisture_level",
    "storage_type",
    "customers_per_day",
    "flies_present",
]

# Numeric / continuous features
num_cols = [
    "hours_per_day",
    "ambient_temperature_c",
    "relative_humidity_pct",
    "time_since_preparation_hours",
    "stall_temperature_c",
    "food_core_temp_c",
    "surface_aw_index",
    "distance_to_drain_m",
    "cleaning_frequency_per_day",
]

X_cat = df[cat_cols].astype(str)
X_num = df[num_cols]

y = df["risk_label"]

# 3. Encode categoricals
enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
X_cat_enc = enc.fit_transform(X_cat)

X = np.hstack([X_num.values, X_cat_enc])

# 4. Train / test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

# 5. Train model
clf = RandomForestClassifier(
    n_estimators=400,
    random_state=42,
    class_weight="balanced",
    max_depth=None
)
clf.fit(X_train, y_train)

# 6. Evaluation
y_pred = clf.predict(X_test)

print("\nClassification report:\n")
print(classification_report(y_test, y_pred))

print("Confusion matrix:\n")
print(confusion_matrix(y_test, y_pred))

# 7. Save model + encoder + feature order
feature_order = cat_cols + num_cols

joblib.dump(clf, "foodguardml_model_expanded.pkl")
joblib.dump(enc, "foodguardml_encoder_expanded.pkl")
joblib.dump(feature_order, "foodguardml_feature_order_expanded.pkl")

print("\nSaved:")
print(" - foodguardml_model_expanded.pkl")
print(" - foodguardml_encoder_expanded.pkl")
print(" - foodguardml_feature_order_expanded.pkl")

# 8. Simple prediction demo for one hypothetical vendor
example = {
    "food_type": "Chaat",
    "hours_per_day": 9,
    "customers_per_day": "High",
    "water_source": "Borewell",
    "change_of_oil": "Medium",
    "hand_hygiene": "Medium",
    "utensil_hygiene": "Medium",
    "covered_food": "Low",
    "proximity_to_waste": "High",
    "ambient_temperature_c": 34.0,
    "relative_humidity_pct": 70,
    "surface_moisture_level": "High",
    "time_since_preparation_hours": 4.0,
    "storage_type": "Open",
    "stall_temperature_c": 34.5,
    "food_core_temp_c": 60.0,
    "surface_aw_index": 0.96,
    "distance_to_drain_m": 1.5,
    "flies_present": "Yes",
    "cleaning_frequency_per_day": 2,
}

example_cat = pd.DataFrame([example])[cat_cols].astype(str)
example_num = pd.DataFrame([example])[num_cols]

example_cat_enc = enc.transform(example_cat)
example_X = np.hstack([example_num.values, example_cat_enc])

pred = clf.predict(example_X)[0]
print("\nExample prediction risk_label:", pred)
