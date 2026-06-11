import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

RANDOM_STATE = 42

# ── Train model on full dataset ───────────────────────────────────────────────
df = pd.read_csv("heart (1).csv")

# Outlier removal
numeric_cols = ["Age", "RestingBP", "Cholesterol", "MaxHR", "Oldpeak"]
df_clean = df.copy()
for col in numeric_cols:
    Q1, Q3 = df_clean[col].quantile([0.25, 0.75])
    IQR = Q3 - Q1
    df_clean = df_clean[
        (df_clean[col] >= Q1 - 1.5 * IQR) &
        (df_clean[col] <= Q3 + 1.5 * IQR)
    ]

# Store medians/modes for filling missing user inputs
medians = {
    "Cholesterol":  int(df_clean["Cholesterol"].median()),
    "FastingBS":    int(df_clean["FastingBS"].mode()[0]),
    "MaxHR":        int(df_clean["MaxHR"].median()),
    "Oldpeak":      round(float(df_clean["Oldpeak"].median()), 1),
    "RestingECG":   df_clean["RestingECG"].mode()[0],
    "ExerciseAngina": df_clean["ExerciseAngina"].mode()[0],
    "ST_Slope":     df_clean["ST_Slope"].mode()[0],
    "ChestPainType": df_clean["ChestPainType"].mode()[0],
}

cat_cols = ["Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope"]
df_encoded = pd.get_dummies(df_clean, columns=cat_cols, drop_first=True)
feature_cols = [c for c in df_encoded.columns if c != "HeartDisease"]

X = df_encoded[feature_cols]
y = df_encoded["HeartDisease"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE
)
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
rf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
rf.fit(X_train_sc, y_train)

# ── Collect user inputs ───────────────────────────────────────────────────────
print("=" * 45)
print("   Heart Disease Risk Checker")
print("=" * 45)

age = int(input("Age                     : "))

sex_input = input("Gender (M / F)           : ").strip().upper()
while sex_input not in ("M", "F"):
    sex_input = input("  Please enter M or F   : ").strip().upper()

bp = int(input("Resting Blood Pressure   : "))

cholesterol = int(input("Cholesterol (mg/dL)      : "))

print("Resting ECG options      : Normal | ST | LVH")
ecg_input = input("Resting ECG              : ").strip()
while ecg_input not in ("Normal", "ST", "LVH"):
    ecg_input = input("  Enter Normal, ST or LVH: ").strip()

# Weight note — not a feature in the dataset
weight_kg = input("Weight (kg, optional)    : ").strip()
if weight_kg:
    print(f"  Note: weight ({weight_kg} kg) is logged but not used by this model's features.")

# ── Build input row using user values + dataset medians ───────────────────────
raw = {
    "Age":            age,
    "Sex":            sex_input,
    "ChestPainType":  medians["ChestPainType"],
    "RestingBP":      bp,
    "Cholesterol":    cholesterol,
    "FastingBS":      medians["FastingBS"],
    "RestingECG":     ecg_input,
    "MaxHR":          medians["MaxHR"],
    "ExerciseAngina": medians["ExerciseAngina"],
    "Oldpeak":        medians["Oldpeak"],
    "ST_Slope":       medians["ST_Slope"],
}

input_df = pd.DataFrame([raw])
input_encoded = pd.get_dummies(input_df, columns=cat_cols, drop_first=True)

# Align columns with training data
for col in feature_cols:
    if col not in input_encoded.columns:
        input_encoded[col] = 0
input_encoded = input_encoded[feature_cols]

input_sc = scaler.transform(input_encoded)

# ── Predict ───────────────────────────────────────────────────────────────────
prob = rf.predict_proba(input_sc)[0][1]
risk_pct = round(prob * 100, 1)

print()
print("=" * 45)
print(f"  Predicted Heart Disease Risk: {risk_pct}%")

if risk_pct < 30:
    level = "LOW RISK"
elif risk_pct < 60:
    level = "MODERATE RISK"
else:
    level = "HIGH RISK"

print(f"  Risk Level: {level}")
print("=" * 45)
print()
print("  Disclaimer: This is a statistical estimate")
print("  based on population data. Consult a doctor")
print("  for a proper medical assessment.")
print("=" * 45)
