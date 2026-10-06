"""Entraîne le meilleur modèle (SVM linéaire) + K-Means et sauvegarde les fichiers pour Streamlit."""
import json, joblib, pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report

scale_cols = ["age_num","time_in_hospital","num_lab_procedures","num_procedures","num_medications",
              "number_outpatient","number_emergency","number_inpatient","number_diagnoses",
              "n_drugs","n_drug_changes","total_prior_visits"]
FEATURES = ["age_num","time_in_hospital","num_lab_procedures","num_procedures","num_medications",
            "number_diagnoses","n_drugs","n_drug_changes","number_outpatient","number_emergency","number_inpatient"]

# ---------- 1. Classification ----------
d = pd.read_csv("diabetic_data_classification.csv")
X, y = d.drop(columns="readmit_30"), d["readmit_30"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

prep = ColumnTransformer([("num", StandardScaler(), scale_cols)], remainder="passthrough")
pipe = Pipeline([("preparation", prep), ("svm", LinearSVC(dual=False, max_iter=10000, random_state=42))])
grid = GridSearchCV(pipe, {"svm__C": [0.01, 0.1, 1, 10], "svm__class_weight": [None, "balanced"]},
                    scoring="f1", cv=StratifiedKFold(3, shuffle=True, random_state=42), n_jobs=2)
grid.fit(X_train, y_train)
print("Meilleurs paramètres :", grid.best_params_, "| F1 CV :", round(grid.best_score_, 3))
best = grid.best_estimator_
print(classification_report(y_test, best.predict(X_test), digits=3))

# probabilité calibrée (LinearSVC n'en fournit pas) -> utilisée pour afficher un "risque en %"
calib = CalibratedClassifierCV(best, cv=3, method="sigmoid").fit(X_train, y_train)

joblib.dump(best, "model_svm.joblib")
joblib.dump(calib, "model_svm_calibre.joblib")
json.dump(list(X.columns), open("model_columns.json", "w"), ensure_ascii=False)

# ---------- 2. Segmentation ----------
seg = pd.read_csv("diabetic_data_clean.csv")
scaler = StandardScaler().fit(seg[FEATURES])
km = KMeans(n_clusters=4, init="k-means++", n_init=50, random_state=42).fit(scaler.transform(seg[FEATURES]))
joblib.dump(scaler, "scaler_segmentation.joblib"); joblib.dump(km, "kmeans_segmentation.joblib")
seg["segment"] = km.labels_
print(seg.groupby("segment")["readmit_30"].agg(["size", "mean"]).round(3))
# profil moyen (valeurs brutes) pour nommer les segments dans l'app
import numpy as np
raw = seg[FEATURES].copy()
for c in ["num_medications","number_outpatient","number_emergency","number_inpatient","time_in_hospital","num_procedures","n_drug_changes"]:
    raw[c] = np.expm1(raw[c])
print(raw.groupby(seg["segment"]).mean().round(2).T)
