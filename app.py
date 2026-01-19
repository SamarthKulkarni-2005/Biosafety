# app.py  -- FoodGuardML Flask app with REWEIGHTED scoring
from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib

app = Flask(__name__)

# Load model, encoder, feature order
model = joblib.load("foodguardml_model_expanded.pkl")
encoder = joblib.load("foodguardml_encoder_expanded.pkl")
feature_order = joblib.load("foodguardml_feature_order_expanded.pkl")

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


def compute_risk_score(data: dict) -> float:
    """
    Reweighted scoring with HIGH emphasis on:
    - Temperature & humidity (dominant spoilage drivers)
    - Water source (contamination risk)
    - Oil change frequency (oxidation & rancidity)
    - Hygiene (hand & utensil)
    - Covered food
    - Proximity to waste (cross-contamination)
    """
    score = 0.0

    t = data["ambient_temperature_c"]
    h = data["relative_humidity_pct"]
    hold = data["time_since_preparation_hours"]
    core_t = data["food_core_temp_c"]
    aw = data["surface_aw_index"]
    dist = data["distance_to_drain_m"]
    cfreq = data["cleaning_frequency_per_day"]

    # ===== 1. TEMPERATURE (HEAVY WEIGHT) =====
    if t >= 45:
        score += 8
    elif t >= 40:
        score += 6
    elif t >= 35:
        score += 5
    elif t >= 30:
        score += 4
    elif t >= 25:
        score += 2
    else:
        score += 0

    # ===== 2. HUMIDITY (HEAVY WEIGHT) =====
    if h >= 90:
        score += 7
    elif h >= 80:
        score += 5
    elif h >= 70:
        score += 4
    elif h >= 60:
        score += 2
    else:
        score += 0

    # ===== 3. TIME-TEMP INTERACTION =====
    if t >= 35 and hold >= 4:
        score += 3
    if t >= 40 and hold >= 6:
        score += 4

    # ===== 4. HOLDING TIME =====
    if hold >= 8:
        score += 5
    elif hold >= 6:
        score += 3
    elif hold >= 4:
        score += 2
    elif hold >= 2:
        score += 1

    # ===== 5. WATER SOURCE (HIGH WEIGHT) =====
    ws = data["water_source"]
    if ws == "Can":
        score += 5
    elif ws == "Borewell":
        score += 3
    # Tap = 0 (assumed treated)

    # ===== 6. OIL CHANGE (HIGH WEIGHT) =====
    co = data["change_of_oil"]
    if co == "Low":
        score += 5
    elif co == "Medium":
        score += 2
    # High = 0 (regularly changed)

    # ===== 7. HAND HYGIENE (HIGH WEIGHT) =====
    hh = data["hand_hygiene"]
    if hh == "Low":
        score += 5
    elif hh == "Medium":
        score += 2
    # High = 0

    # ===== 8. UTENSIL HYGIENE (HIGH WEIGHT) =====
    uh = data["utensil_hygiene"]
    if uh == "Low":
        score += 5
    elif uh == "Medium":
        score += 2
    # High = 0

    # ===== 9. COVERED FOOD (HIGH WEIGHT) =====
    cf = data["covered_food"]
    if cf == "Low":
        score += 6
    elif cf == "Medium":
        score += 3
    # High = 0

    # ===== 10. PROXIMITY TO WASTE (VERY HIGH WEIGHT) =====
    pw = data["proximity_to_waste"]
    if pw == "High" or dist <= 1.5:
        score += 8
    elif pw == "Medium" or dist <= 4:
        score += 4
    elif dist <= 8:
        score += 1
    # Low / far = 0

    # ===== 11. FLIES / VECTORS =====
    if data["flies_present"] == "Yes":
        score += 4
        if pw == "High" or dist <= 2:
            score += 2

    # ===== 12. CORE FOOD TEMP =====
    if 15 <= core_t <= 55:
        score += 3
    elif core_t < 5:
        score -= 1
    elif core_t >= 65:
        score -= 1

    # ===== 13. SURFACE AW / MOISTURE =====
    if aw >= 0.97:
        score += 2
    elif aw >= 0.93:
        score += 1

    # ===== 14. CLEANING FREQUENCY =====
    if cfreq <= 1:
        score += 2
    elif cfreq <= 3:
        score += 1
    elif cfreq >= 6:
        score -= 1

    # ===== 15. FOOD TYPE & TRAFFIC (MINOR) =====
    if data["food_type"] in ["Chaat", "Juice"]:
        score += 1
    if data["customers_per_day"] == "High":
        score += 1

    return max(score, 0.0)


def map_score_to_label(score: float) -> str:
    """
    Map score to Low / Medium / High with emphasis on these factors.
    """
    if score <= 15:
        return "Low"
    elif score <= 32:
        return "Medium"
    else:
        return "High"


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    # 1. Collect form data
    data = {k: v for k, v in request.form.items()}

    # Convert numeric fields
    num_fields = [
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
    for col in num_fields:
        data[col] = float(data[col])

    # 2. Rule-based risk
    rule_score = compute_risk_score(data)
    rule_label = map_score_to_label(rule_score)

    # 3. ML refinement only for Medium
    if rule_label == "Medium":
        cat_df = pd.DataFrame([data])[cat_cols].astype(str)
        num_df = pd.DataFrame([data])[num_cols]
        cat_enc = encoder.transform(cat_df)
        X = np.hstack([num_df.values, cat_enc])
        ml_label = model.predict(X)[0]
        final_label = ml_label
    else:
        final_label = rule_label

    # 4. Message
    if final_label == "High":
        msg = (
            "High risk: critical factors for spoilage present (high temperature/humidity, poor water quality, "
            "unsafe hygiene, uncovered food, proximity to waste). Immediate corrective actions required."
        )
    elif final_label == "Medium":
        msg = (
            "Medium risk: some concerning factors present (elevated temperature, humidity, or hygiene issues). "
            "Improve water quality, increase oil change frequency, improve hygiene and covering, increase distance from waste."
        )
    else:
        msg = (
            "Low risk: conditions are safe, but maintain strict hygiene practices, regular cleaning, "
            "proper water source, and monitor temperature/humidity continuously."
        )

    return jsonify({
        "risk_label": final_label,
        "rule_score": round(rule_score, 1),
        "rule_label": rule_label,
        "message": msg,
    })


if __name__ == "__main__":
    app.run(debug=True)
#https://github.com/SamarthKulkarni-2005/Biosafety.git