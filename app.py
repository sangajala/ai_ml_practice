from flask import Flask, request, jsonify, render_template
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

RANDOM_STATE = 42

# ── Train model at startup ────────────────────────────────────────────────────
df = pd.read_csv("heart (1).csv")

numeric_cols = ["Age", "RestingBP", "Cholesterol", "MaxHR", "Oldpeak"]
df_clean = df.copy()
for col in numeric_cols:
    Q1, Q3 = df_clean[col].quantile([0.25, 0.75])
    IQR = Q3 - Q1
    df_clean = df_clean[
        (df_clean[col] >= Q1 - 1.5 * IQR) &
        (df_clean[col] <= Q3 + 1.5 * IQR)
    ]

medians = {
    "Cholesterol":    int(df_clean["Cholesterol"].median()),
    "FastingBS":      int(df_clean["FastingBS"].mode()[0]),
    "MaxHR":          int(df_clean["MaxHR"].median()),
    "Oldpeak":        round(float(df_clean["Oldpeak"].median()), 1),
    "ExerciseAngina": df_clean["ExerciseAngina"].mode()[0],
    "ST_Slope":       df_clean["ST_Slope"].mode()[0],
    "ChestPainType":  df_clean["ChestPainType"].mode()[0],
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

print("Model trained and ready.")


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("home.html", active="home")


@app.route("/heart-risk")
def heart_risk():
    return render_template("heart_risk.html", active="heart-risk")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    try:
        age         = int(data["age"])
        sex         = data["sex"]
        bp          = int(data["bp"])
        cholesterol = int(data["cholesterol"])
        ecg         = data["ecg"]
    except (KeyError, ValueError) as e:
        return jsonify({"error": f"Invalid input: {e}"}), 400

    raw = {
        "Age":            age,
        "Sex":            sex,
        "ChestPainType":  medians["ChestPainType"],
        "RestingBP":      bp,
        "Cholesterol":    cholesterol,
        "FastingBS":      medians["FastingBS"],
        "RestingECG":     ecg,
        "MaxHR":          medians["MaxHR"],
        "ExerciseAngina": medians["ExerciseAngina"],
        "Oldpeak":        medians["Oldpeak"],
        "ST_Slope":       medians["ST_Slope"],
    }

    input_df = pd.DataFrame([raw])
    input_encoded = pd.get_dummies(input_df, columns=cat_cols, drop_first=True)
    for col in feature_cols:
        if col not in input_encoded.columns:
            input_encoded[col] = 0
    input_encoded = input_encoded[feature_cols]

    input_sc = scaler.transform(input_encoded)
    prob = rf.predict_proba(input_sc)[0][1]
    risk_pct = round(prob * 100, 1)

    if risk_pct < 30:
        level = "Low Risk"
    elif risk_pct < 60:
        level = "Moderate Risk"
    else:
        level = "High Risk"

    return jsonify({"risk": risk_pct, "level": level})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", debug=False, port=port)
