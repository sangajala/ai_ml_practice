from flask import Flask, request, jsonify, render_template
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier

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
    "RestingECG":     df_clean["RestingECG"].mode()[0],
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

knn_clf = KNeighborsClassifier(n_neighbors=5)
knn_clf.fit(X_train_sc, y_train)

# ── Unsupervised models (no labels) ──────────────────────────────────────────
X_all = df_encoded[feature_cols]
uns_scaler = StandardScaler()
X_all_sc = uns_scaler.fit_transform(X_all)

kmeans = KMeans(n_clusters=3, random_state=RANDOM_STATE, n_init=10)
kmeans.fit(X_all_sc)

# Cluster risk labels derived from HeartDisease mean per cluster
cluster_hd = {}
for c in range(3):
    mask = kmeans.labels_ == c
    cluster_hd[c] = df_encoded.loc[X_all.index[mask], "HeartDisease"].mean()

# Map cluster id → risk label (low/moderate/high)
def cluster_risk_label(cid):
    rate = cluster_hd[cid]
    if rate < 0.30:
        return "Low Risk Group", rate
    elif rate < 0.65:
        return "Moderate Risk Group", rate
    else:
        return "High Risk Group", rate

# KNN anomaly detector — fit on full scaled data
knn_detector = NearestNeighbors(n_neighbors=6)
knn_detector.fit(X_all_sc)
all_distances, _ = knn_detector.kneighbors(X_all_sc)
knn_all_scores = all_distances[:, 1:].mean(axis=1)
anomaly_threshold = float(np.percentile(knn_all_scores, 95))

print("Models trained and ready.")


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


@app.route("/pattern-analysis")
def pattern_analysis():
    return render_template("pattern_analysis.html", active="pattern-analysis")


@app.route("/predict-pattern", methods=["POST"])
def predict_pattern():
    data = request.get_json()

    try:
        age     = int(data["age"])
        oldpeak = float(data["oldpeak"])
    except (KeyError, ValueError) as e:
        return jsonify({"error": f"Invalid input: {e}"}), 400

    raw = {
        "Age":            age,
        "Sex":            medians.get("Sex", "M"),
        "ChestPainType":  medians["ChestPainType"],
        "RestingBP":      int(df_clean["RestingBP"].median()),
        "Cholesterol":    medians["Cholesterol"],
        "FastingBS":      medians["FastingBS"],
        "RestingECG":     medians["RestingECG"],
        "MaxHR":          medians["MaxHR"],
        "ExerciseAngina": medians["ExerciseAngina"],
        "Oldpeak":        oldpeak,
        "ST_Slope":       medians["ST_Slope"],
    }

    input_df = pd.DataFrame([raw])
    input_encoded = pd.get_dummies(input_df, columns=cat_cols, drop_first=True)
    for col in feature_cols:
        if col not in input_encoded.columns:
            input_encoded[col] = 0
    input_encoded = input_encoded[feature_cols]

    input_sc = uns_scaler.transform(input_encoded)

    # Cluster assignment
    cluster_id = int(kmeans.predict(input_sc)[0])
    risk_label, hd_rate = cluster_risk_label(cluster_id)

    # KNN anomaly score
    distances, _ = knn_detector.kneighbors(input_sc)
    knn_score = float(distances[0, 1:].mean())
    is_anomaly = knn_score > anomaly_threshold
    anomaly_pct = round(min(knn_score / anomaly_threshold * 100, 200), 1)

    # Nearest neighbours indices for context
    _, indices = knn_detector.kneighbors(input_sc, n_neighbors=4)
    neighbours = df_clean.iloc[indices[0][1:]].copy()
    neighbour_hd_rate = round(float(df_encoded.loc[
        df_clean.index[indices[0][1:]], "HeartDisease"
    ].mean() * 100), 1)

    return jsonify({
        "cluster_id":       cluster_id,
        "risk_label":       risk_label,
        "hd_rate":          round(hd_rate * 100, 1),
        "knn_score":        round(knn_score, 3),
        "threshold":        round(anomaly_threshold, 3),
        "is_anomaly":       bool(is_anomaly),
        "anomaly_pct":      anomaly_pct,
        "neighbour_hd_pct": neighbour_hd_rate,
    })


@app.route("/quick-check")
def quick_check():
    return render_template("quick_check.html", active="quick-check")


@app.route("/predict-quick", methods=["POST"])
def predict_quick():
    data = request.get_json()
    try:
        age = int(data["age"])
        bp  = int(data["bp"])
    except (KeyError, ValueError) as e:
        return jsonify({"error": f"Invalid input: {e}"}), 400

    raw = {
        "Age":            age,
        "Sex":            "M",
        "ChestPainType":  medians["ChestPainType"],
        "RestingBP":      bp,
        "Cholesterol":    medians["Cholesterol"],
        "FastingBS":      medians["FastingBS"],
        "RestingECG":     medians["RestingECG"],
        "MaxHR":          medians["MaxHR"],
        "ExerciseAngina": medians["ExerciseAngina"],
        "Oldpeak":        medians["Oldpeak"],
        "ST_Slope":       medians["ST_Slope"],
    }

    input_df      = pd.DataFrame([raw])
    input_encoded = pd.get_dummies(input_df, columns=cat_cols, drop_first=True)
    for col in feature_cols:
        if col not in input_encoded.columns:
            input_encoded[col] = 0
    input_encoded = input_encoded[feature_cols]

    input_sc  = scaler.transform(input_encoded)
    prob      = knn_clf.predict_proba(input_sc)[0][1]
    risk_pct  = round(prob * 100, 1)

    if risk_pct < 30:
        level = "Low Risk"
    elif risk_pct < 60:
        level = "Moderate Risk"
    else:
        level = "High Risk"

    # BP category for context
    if bp < 120:
        bp_cat = "Normal"
    elif bp < 130:
        bp_cat = "Elevated"
    elif bp < 140:
        bp_cat = "High Stage 1"
    else:
        bp_cat = "High Stage 2"

    return jsonify({"risk": risk_pct, "level": level, "bp_category": bp_cat})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", debug=False, port=port)
