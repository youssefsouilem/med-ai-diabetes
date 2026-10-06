import json
import os
import textwrap

import joblib
import numpy as np
import pandas as pd
import streamlit as st


# ================================================================
# PAGE CONFIG
# ================================================================

st.set_page_config(
    page_title="Diabetes Readmission AI",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ================================================================
# HTML HELPER
# ================================================================

def html(content: str):
    """
    Render multiline HTML without Markdown mistaking indentation
    or blank lines for a code block: every line is stripped and
    empty lines are removed.
    """
    cleaned = "\n".join(
        line.strip()
        for line in textwrap.dedent(content).splitlines()
        if line.strip()
    )
    st.markdown(cleaned, unsafe_allow_html=True)


# ================================================================
# CUSTOM CSS
# ================================================================

st.markdown(
    textwrap.dedent(
        """
        <style>

        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Playfair+Display:wght@600;700&display=swap');

        /* =========================================================
           GLOBAL
           ========================================================= */

        html,
        body,
        [class*="css"] {
            font-family: 'DM Sans', sans-serif;
        }

        .stApp {
            background: #F5F0EA;
            color: #403832;
        }

        .block-container {
            max-width: 1450px;
            padding-top: 2rem;
            padding-bottom: 4rem;
        }

        #MainMenu {
            visibility: hidden;
        }

        footer {
            visibility: hidden;
        }

        header {
            background: transparent !important;
        }


        /* =========================================================
           SIDEBAR
           ========================================================= */

        section[data-testid="stSidebar"] {
            background: #E8DDD2;
            border-right: 1px solid #D6C6B8;
        }

        section[data-testid="stSidebar"] > div {
            padding-top: 1.8rem;
        }

        .sidebar-logo {
            color: #4B4038;
            font-size: 1.6rem;
            font-weight: 700;
            letter-spacing: -0.03em;
        }

        .sidebar-subtitle {
            color: #81736A;
            font-size: 0.8rem;
            margin-top: 0.15rem;
            margin-bottom: 1.8rem;
        }

        .sidebar-card {
            background: rgba(255, 250, 246, 0.65);
            border: 1px solid #D7C8BA;
            border-radius: 17px;
            padding: 1rem;
            margin-bottom: 0.8rem;
            box-shadow: 0 4px 15px rgba(91, 72, 58, 0.035);
        }

        .sidebar-label {
            color: #8B7A6E;
            font-size: 0.66rem;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            font-weight: 700;
        }

        .sidebar-value {
            color: #4B4038;
            font-size: 0.9rem;
            font-weight: 600;
            margin-top: 0.35rem;
        }

        .sidebar-footer {
            color: #86776D;
            font-size: 0.72rem;
            line-height: 1.6;
            margin-top: 2rem;
        }


        /* =========================================================
           HERO
           ========================================================= */

        .hero {
            background:
                linear-gradient(
                    135deg,
                    #E8D7C8 0%,
                    #F3E9E0 52%,
                    #E6D6C7 100%
                );

            border: 1px solid #D9C8BA;
            border-radius: 28px;
            padding: 2.6rem 2.8rem;
            margin-bottom: 2rem;
            box-shadow:
                0 18px 45px rgba(89, 70, 56, 0.08);

            position: relative;
            overflow: hidden;
        }

        .hero::after {
            
            position: absolute;
            right: 45px;
            top: 18px;
            font-size: 6rem;
            color: rgba(112, 92, 77, 0.07);
        }

        .eyebrow {
            color: #6F8570;
            text-transform: uppercase;
            letter-spacing: 0.13em;
            font-size: 0.7rem;
            font-weight: 700;
            margin-bottom: 0.8rem;
        }

        .hero-title {
            font-family: 'Playfair Display', serif;
            color: #403731;
            font-size: 2.65rem;
            line-height: 1.12;
            margin: 0;
            position: relative;
            z-index: 2;
        }

        .hero-description {
            color: #76695F;
            font-size: 0.96rem;
            line-height: 1.7;
            max-width: 760px;
            margin-top: 0.9rem;
            position: relative;
            z-index: 2;
        }

        .hero-badge {
            display: inline-block;
            background: rgba(255, 255, 255, 0.55);
            border: 1px solid rgba(143, 119, 100, 0.18);
            color: #66564A;
            padding: 0.45rem 0.8rem;
            border-radius: 999px;
            font-size: 0.72rem;
            font-weight: 600;
            margin-top: 1rem;
            position: relative;
            z-index: 2;
        }


        /* =========================================================
           SECTION HEADERS
           ========================================================= */

        .section-title {
            color: #4B4038;
            font-size: 1.15rem;
            font-weight: 700;
            margin-top: 1.5rem;
            margin-bottom: 0.2rem;
        }

        .section-description {
            color: #897A70;
            font-size: 0.8rem;
            margin-bottom: 1rem;
        }

            /* =========================================================
            INPUTS
            ========================================================= */

            .stSelectbox label,
            .stSlider label,
            .stNumberInput label {
                color: #62564E !important;
                font-size: 0.8rem !important;
                font-weight: 600 !important;
            }


            /* SELECT BOX */

            div[data-testid="stSelectbox"] div[data-baseweb="select"] > div {
                background-color: #758A73 !important;
                border: 1px solid #667A65 !important;
                border-radius: 12px !important;
                color: #F8F5F2 !important;
            }

            div[data-testid="stSelectbox"] div[data-baseweb="select"] span {
                color: #F8F5F2 !important;
                font-weight: 500 !important;
            }

            div[data-testid="stSelectbox"] div[data-baseweb="select"] svg {
                fill: #F8F5F2 !important;
                color: #F8F5F2 !important;
            }


            /* NUMBER INPUT */

            div[data-testid="stNumberInput"] input {
                background: #FCFAF8 !important;
                border-color: #D8CBC0 !important;
                border-radius: 11px !important;
                color: #403832 !important;
            }


            /* SLIDER */

            div[data-testid="stSlider"] div[role="slider"] {
                background-color: #4F453F !important;
                border: 2px solid #4F453F !important;
                box-shadow: none !important;
            }

            div[data-testid="stSliderThumbValue"],
            div[data-testid="stSliderThumbValue"] * {
                color: #4F453F !important;
            }

        /* =========================================================
           SUBMIT BUTTON
           ========================================================= */

        .stFormSubmitButton > button {
            width: 100%;
            min-height: 54px;

            background: #7F9279;
            box-shadow: 0 9px 22px rgba(127, 146, 121, 0.28);

            border: none;
            border-radius: 15px;

            font-size: 0.9rem;
            font-weight: 700;

            box-shadow:
                0 9px 22px rgba(112, 92, 77, 0.2);

            transition: all 0.2s ease;
        }

        .stFormSubmitButton > button:hover {
            background: #5E7360;
            box-shadow: 0 12px 28px rgba(94, 115, 96, 0.3);

            transform: translateY(-1px);

            box-shadow:
                0 12px 28px rgba(112, 92, 77, 0.25);
        }


        /* =========================================================
           RESULTS
           ========================================================= */

        .results-header {
            margin-top: 2.4rem;
            margin-bottom: 1.2rem;
        }

        .results-title {
            font-family: 'Playfair Display', serif;
            color: #433932;
            font-size: 2rem;
            margin: 0;
        }

        .result-card {
            background: #FCFAF8;
            border: 1px solid #E0D4CA;
            border-radius: 21px;
            padding: 1.35rem;
            min-height: 145px;
            box-shadow:
                0 9px 28px rgba(95, 76, 60, 0.05);
        }

        .result-label {
            color: #8A7B70;
            font-size: 0.68rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            font-weight: 700;
        }

        .result-value {
            color: #433932;
            font-size: 1.45rem;
            font-weight: 700;
            margin-top: 0.6rem;
            line-height: 1.25;
        }

        .result-small {
            color: #8A7B70;
            font-size: 0.75rem;
            margin-top: 0.35rem;
            line-height: 1.4;
        }


        /* =========================================================
           RISK
           ========================================================= */

        .risk-low {
            background: #E9F0E9;
            border: 1px solid #C9D9CA;
            color: #4C6851;

            border-radius: 19px;
            padding: 1.2rem 1.35rem;
            margin-top: 1rem;
        }

        .risk-high {
            background: #F3E5DF;
            border: 1px solid #DFC3B7;
            color: #825548;

            border-radius: 19px;
            padding: 1.2rem 1.35rem;
            margin-top: 1rem;
        }

        .risk-title {
            font-weight: 700;
            font-size: 0.95rem;
            margin-bottom: 0.3rem;
        }

        .risk-text {
            font-size: 0.8rem;
            line-height: 1.6;
        }


        /* =========================================================
           SEGMENT
           ========================================================= */

        .segment-card {
            background: #E4EBE2;
            border: 1px solid #DCCFC3;
            border-radius: 20px;

            padding: 1.35rem;
            margin-top: 1rem;
        }

        .segment-number {
            color: #6F8570;
            font-size: 0.67rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            font-weight: 700;
        }

        .segment-name {
            color: #493F38;
            font-size: 1.1rem;
            font-weight: 700;
            margin-top: 0.4rem;
            margin-bottom: 0.3rem;
        }

        .segment-description {
            color: #75685E;
            font-size: 0.8rem;
            line-height: 1.6;
        }


        /* =========================================================
           PROGRESS
           ========================================================= */

        div[data-testid="stProgressBar"] {
            margin-top: 0.25rem;
        }

        div[data-testid="stProgressBar"] > div {
            background-color: #7F9279;
            border-radius: 999px;
        }

        div[data-testid="stProgressBar"] > div > div {
            background-color: #806B5A;
            border-radius: 999px;
        }


        /* =========================================================
           EXPANDER
           ========================================================= */

        div[data-testid="stExpander"] {
            background: #FCFAF8;
            border: 1px solid #DDD0C5;
            border-radius: 16px;
            overflow: hidden;
        }


        /* =========================================================
           DATAFRAME
           ========================================================= */

        div[data-testid="stDataFrame"] {
            border-radius: 12px;
            overflow: hidden;
        }


        /* =========================================================
           DISCLAIMER
           ========================================================= */

        .disclaimer {
            background: #EEE8E2;
            border: 1px solid #DDD2C9;
            border-radius: 16px;

            padding: 0.9rem 1.1rem;
            margin-top: 2rem;

            color: #766A61;
            font-size: 0.73rem;
            line-height: 1.55;
        }


        /* =========================================================
           MOBILE
           ========================================================= */

        @media (max-width: 768px) {

            .block-container {
                padding: 1rem;
            }

            .hero {
                padding: 1.7rem;
                border-radius: 20px;
            }

            .hero-title {
                font-size: 2rem;
            }

            .hero-description {
                font-size: 0.88rem;
            }
        }
        /* Top toolbar text */
[data-testid="stToolbar"] *,
header * {
    color: #6F8570 !important;   /* Nude green */
}

/* Toolbar icons */
[data-testid="stToolbar"] svg {
    fill: #6F8570 !important;
}

        </style>
        """
    ),
    unsafe_allow_html=True
)


# ================================================================
# MODEL FILES
# ================================================================

MODEL_FILES = {
    "svm": "model_svm.joblib",
    "calib": "model_svm_calibre.joblib",
    "scaler": "scaler_segmentation.joblib",
    "km": "kmeans_segmentation.joblib",
    "cols": "model_columns.json",
    "meta": "prep_meta.json",
}


# ================================================================
# SEGMENTATION
# ================================================================

SEG_FEATURES = [
    "age_num",
    "time_in_hospital",
    "num_lab_procedures",
    "num_procedures",
    "num_medications",
    "number_diagnoses",
    "n_drugs",
    "n_drug_changes",
    "number_outpatient",
    "number_emergency",
    "number_inpatient",
]


SEGMENTS = {
    0: (
        "Forts utilisateurs des urgences",
        "Nombreux passages aux urgences et hospitalisations antérieures."
    ),
    1: (
        "Âgés, séjours longs, traitement stable",
        "Patients plus âgés, séjour long et traitement relativement stable."
    ),
    2: (
        "Traitement complexe ou instable",
        "Nombreux médicaments et changements de traitement."
    ),
    3: (
        "Séjours courts, profil léger",
        "Patients plus jeunes, séjour court et faible charge médicamenteuse."
    ),
}


# ================================================================
# LOAD MODELS
# ================================================================

@st.cache_resource
def load_all():

    missing = [
        filename
        for filename in MODEL_FILES.values()
        if not os.path.exists(filename)
    ]

    if missing:
        raise FileNotFoundError(
            "Fichiers manquants : " + ", ".join(missing)
        )

    with open(MODEL_FILES["cols"], "r", encoding="utf-8") as f:
        cols = json.load(f)

    with open(MODEL_FILES["meta"], "r", encoding="utf-8") as f:
        meta = json.load(f)

    return {
        "svm": joblib.load(MODEL_FILES["svm"]),
        "calib": joblib.load(MODEL_FILES["calib"]),
        "scaler": joblib.load(MODEL_FILES["scaler"]),
        "km": joblib.load(MODEL_FILES["km"]),
        "cols": cols,
        "meta": meta,
    }


try:
    M = load_all()

except Exception as e:

    st.error("Impossible de charger les modèles.")

    st.info(
        "Vérifiez que les fichiers suivants sont "
        "dans le même dossier que app.py :"
    )

    for filename in MODEL_FILES.values():
        st.code(filename)

    with st.expander("Détails techniques"):
        st.code(str(e))

    st.stop()


# ================================================================
# METADATA
# ================================================================

CAPS = M["meta"].get("caps", {})
DRUGS = M["meta"].get("drug_cols", [])
MODEL_COLS = M["cols"]


def default_index(options):
    """Index of the 'unknown' category if present, otherwise 0."""
    for unknown in ("Inconnu", "Unknown", "?"):
        if unknown in options:
            return options.index(unknown)
    return 0


# ================================================================
# FEATURE ENGINEERING
# ================================================================

def build_features(inp: dict):
    """
    Returns (X, num):
      X   -> DataFrame with every model column
      num -> engineered numeric values (after log transform),
             used as fallback for segmentation features
    """

    # Initialize every model column
    row = {column: 0 for column in MODEL_COLS}

    # Variables subject to clipping
    clip_features = [
        "num_lab_procedures",
        "num_medications",
        "number_outpatient",
        "number_emergency",
        "number_inpatient",
        "time_in_hospital",
        "num_procedures",
        "number_diagnoses",
    ]

    v = {}

    for key in clip_features:

        try:
            value = float(inp.get(key, 0))
        except (TypeError, ValueError):
            value = 0.0

        if not np.isfinite(value):
            value = 0.0

        if key in CAPS:
            try:
                value = min(value, float(CAPS[key]))
            except (TypeError, ValueError):
                pass

        v[key] = value

    # Drug variables
    drugs = {
        drug: inp.get("drug_" + drug, "No")
        for drug in DRUGS
    }

    n_drugs = sum(value != "No" for value in drugs.values())
    n_changes = sum(value in ("Up", "Down") for value in drugs.values())

    total_prior = (
        v["number_outpatient"]
        + v["number_emergency"]
        + v["number_inpatient"]
    )

    # Age
    try:
        age = int(inp.get("age", 0))
    except (TypeError, ValueError):
        age = 0

    age = max(0, min(120, age))
    age_num = int(age // 10) * 10 + 5

    # Numerical features
    num = dict(
        v,
        n_drugs=n_drugs,
        n_drug_changes=n_changes,
        total_prior_visits=total_prior,
        age_num=age_num,
    )

    # Log transformation
    for key in M["meta"].get("to_log", []):
        if key in num:
            num[key] = np.log1p(max(0, float(num[key])))

    for key, value in num.items():
        if key in row:
            row[key] = value

    # Binary features
    if "gender" in row:
        row["gender"] = int(inp.get("gender") == "Homme")

    if "change" in row:
        row["change"] = int(n_changes > 0)

    if "diabetesMed" in row:
        row["diabetesMed"] = int(n_drugs > 0)

    # Categorical variables
    categories = {
        "race": inp.get("race"),
        "payer_code": inp.get("payer_code"),
        "medical_specialty": inp.get("medical_specialty"),
        "admission_type_id": inp.get("admission_type_id"),
        "admission_source_id": inp.get("admission_source_id"),
        "discharge_disposition_id": inp.get("discharge_disposition_id"),
        "diag_1_grp": inp.get("diag_1_grp"),
        "diag_2_grp": inp.get("diag_2_grp"),
        "diag_3_grp": inp.get("diag_3_grp"),
        "max_glu_serum": inp.get("max_glu_serum"),
        "A1Cresult": inp.get("A1Cresult"),
        **drugs,
    }

    for name, value in categories.items():

        if value is None:
            continue

        column = f"{name}_{value}"

        if column in row:
            row[column] = 1

    # Final DataFrame
    X = pd.DataFrame([row])
    X = X.reindex(columns=MODEL_COLS, fill_value=0)
    X = X.replace([np.inf, -np.inf], 0)
    X = X.fillna(0)

    return X, num


# ================================================================
# PREDICTION
# ================================================================

def predict_patient(inp):

    X, num = build_features(inp)

    # SVM classification
    svm_prediction = int(M["svm"].predict(X)[0])

    # Calibrated probability
    probabilities = M["calib"].predict_proba(X)
    probability = float(np.clip(float(probabilities[0, 1]), 0.0, 1.0))

    # K-Means segmentation: use the model column when it exists,
    # otherwise fall back to the engineered value (not a silent 0)
    seg_values = {
        f: (X[f].iloc[0] if f in X.columns else num.get(f, 0))
        for f in SEG_FEATURES
    }

    segment_input = pd.DataFrame([seg_values], columns=SEG_FEATURES)
    segment_scaled = M["scaler"].transform(segment_input)
    segment_id = int(M["km"].predict(segment_scaled)[0])

    segment_name, segment_description = SEGMENTS.get(
        segment_id,
        (f"Segment {segment_id}", "Profil non documenté.")
    )

    return {
        "X": X,
        "prediction": svm_prediction,
        "probability": probability,
        "segment_id": segment_id,
        "segment_name": segment_name,
        "segment_description": segment_description,
    }


# ================================================================
# SIDEBAR
# ================================================================

with st.sidebar:

    html(
        """
        <div class="sidebar-logo">MED•AI</div>
        <div class="sidebar-subtitle">Diabetes Readmission Intelligence</div>
        """
    )

    html(
        f"""
        <div class="sidebar-card">
            <div class="sidebar-label">Modèle principal</div>
            <div class="sidebar-value">Linear SVM</div>
        </div>

        <div class="sidebar-card">
            <div class="sidebar-label">Probabilité</div>
            <div class="sidebar-value">SVM calibré · Sigmoïde</div>
        </div>

        <div class="sidebar-card">
            <div class="sidebar-label">Segmentation</div>
            <div class="sidebar-value">K-Means · 4 profils</div>
        </div>

        <div class="sidebar-card">
            <div class="sidebar-label">Variables</div>
            <div class="sidebar-value">{len(MODEL_COLS)} features</div>
        </div>
        """
    )

    st.markdown("---")

    html(
        """
        <div class="sidebar-footer">
            <strong>Dataset</strong><br>
            Diabetes 130-US hospitals<br>
            1999–2008
            <br><br>
            Application pédagogique —
            ne constitue pas un outil
            de diagnostic médical.
        </div>
        """
    )


# ================================================================
# HERO
# ================================================================

html(
    """
    <div class="hero">
        <div class="eyebrow">Clinical Intelligence · Predictive Analytics</div>
        <h1 class="hero-title">
            Réadmission diabète
            <br>
            <span style="color:#806B5A;">à moins de 30 jours</span>
        </h1>
        <div class="hero-description">
            Analysez le profil d'un patient diabétique et estimez
            son risque de réadmission dans les 30 jours suivant
            sa sortie hospitalière.
        </div>
        <div class="hero-badge">✦ IA prédictive · Analyse non diagnostique</div>
    </div>
    """
)


# ================================================================
# PATIENT FORM
# ================================================================

with st.form("patient_form"):

    # ---------------- PATIENT PROFILE ----------------

    html(
        """
        <div class="section-title">Profil du patient</div>
        <div class="section-description">
            Informations générales et caractéristiques du séjour.
        </div>
        """
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        age = st.slider("Âge", min_value=0, max_value=99, value=65)

    with c2:
        gender = st.selectbox("Sexe", ["Femme", "Homme"])

    with c3:
        time_in_hospital = st.slider(
            "Durée du séjour", min_value=1, max_value=14, value=4
        )

    with c4:
        number_diagnoses = st.slider(
            "Nombre de diagnostics", min_value=1, max_value=16, value=7
        )

    # ---------------- HOSPITAL ACTIVITY ----------------

    html(
        """
        <div class="section-title">Activité hospitalière</div>
        <div class="section-description">
            Examens, procédures et charge médicamenteuse.
        </div>
        """
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        num_lab_procedures = st.slider(
            "Analyses laboratoire", min_value=1, max_value=132, value=43
        )

    with c2:
        num_procedures = st.slider(
            "Procédures", min_value=0, max_value=6, value=1
        )

    with c3:
        num_medications = st.slider(
            "Médicaments", min_value=1, max_value=81, value=15
        )

    with c4:
        race = st.selectbox(
            "Origine",
            [
                "Caucasian",
                "AfricanAmerican",
                "Hispanic",
                "Asian",
                "Other",
                "Inconnu",
            ],
        )

    # ---------------- PREVIOUS HEALTHCARE ----------------

    html(
        """
        <div class="section-title">Recours aux soins antérieurs</div>
        <div class="section-description">
            Utilisation des soins durant la période précédente.
        </div>
        """
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        number_outpatient = st.number_input(
            "Consultations externes", min_value=0, max_value=50, value=0
        )

    with c2:
        number_emergency = st.number_input(
            "Passages aux urgences", min_value=0, max_value=50, value=0
        )

    with c3:
        number_inpatient = st.number_input(
            "Hospitalisations", min_value=0, max_value=20, value=0
        )

    # ---------------- ADMISSION & DIAGNOSIS ----------------

    html(
        """
        <div class="section-title">Admission & diagnostics</div>
        <div class="section-description">
            Contexte administratif et médical du séjour.
        </div>
        """
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        admission_type_id = st.selectbox(
            "Type d'admission",
            ["Urgence", "Programmee", "Nouveau-ne", "Inconnu"],
        )

    with c2:
        admission_source_id = st.selectbox(
            "Provenance",
            ["Reference", "Urgences", "Transfert", "Autre", "Inconnu"],
        )

    with c3:
        discharge_disposition_id = st.selectbox(
            "Sortie vers",
            [
                "Domicile",
                "Domicile_soins",
                "Etablissement",
                "Contre_avis",
                "Autre",
                "Inconnu",
            ],
        )

    GRP = [
        "Autre",
        "Circulatoire",
        "Respiratoire",
        "Digestif",
        "Diabete",
        "Blessures",
        "Musculosquelettique",
        "Genito-urinaire",
        "Neoplasmes",
    ]

    c1, c2, c3 = st.columns(3)

    with c1:
        diag_1_grp = st.selectbox("Diagnostic principal", GRP, index=1)

    with c2:
        diag_2_grp = st.selectbox("Diagnostic secondaire", ["Aucun"] + GRP)

    with c3:
        diag_3_grp = st.selectbox("Troisième diagnostic", ["Aucun"] + GRP)

    # ---------------- ADMINISTRATIVE INFORMATION ----------------

    c1, c2 = st.columns(2)

    with c1:
        specialties = M["meta"].get("cats", {}).get(
            "medical_specialty", ["Inconnu"]
        )

        medical_specialty = st.selectbox(
            "Spécialité médicale",
            specialties,
            index=default_index(specialties),
        )

    with c2:
        payers = M["meta"].get("cats", {}).get("payer_code", ["Inconnu"])

        payer_code = st.selectbox(
            "Assurance",
            payers,
            index=default_index(payers),
        )

    # ---------------- EXAMINATIONS ----------------

    html(
        """
        <div class="section-title">Examens & traitements</div>
        <div class="section-description">
            Résultats biologiques et état du traitement.
        </div>
        """
    )

    c1, c2 = st.columns(2)

    with c1:
        A1Cresult = st.selectbox("HbA1c", ["Non testé", "Norm", ">7", ">8"])

    with c2:
        max_glu_serum = st.selectbox(
            "Glycémie sérique maximale",
            ["Non testé", "Norm", ">200", ">300"],
        )

    # ---------------- DRUGS ----------------

    drug_vals = {}

    if DRUGS:

        html(
            """
            <div class="section-description">État des traitements</div>
            """
        )

        drug_columns = st.columns(4)

        for i, drug in enumerate(DRUGS):

            with drug_columns[i % 4]:

                drug_vals[drug] = st.selectbox(
                    drug,
                    ["No", "Steady", "Up", "Down"],
                    key=f"drug_{drug}",
                )

    # ---------------- SUBMIT ----------------

    st.markdown("<br>", unsafe_allow_html=True)

    go = st.form_submit_button(
        "✦  ANALYSER LE PROFIL DU PATIENT",
        use_container_width=True,
    )


# ================================================================
# PREDICTION
# ================================================================

if go:

    patient_input = dict(
        age=age,
        gender=gender,
        time_in_hospital=time_in_hospital,
        number_diagnoses=number_diagnoses,
        num_lab_procedures=num_lab_procedures,
        num_procedures=num_procedures,
        num_medications=num_medications,
        race=race,
        number_outpatient=number_outpatient,
        number_emergency=number_emergency,
        number_inpatient=number_inpatient,
        admission_type_id=admission_type_id,
        admission_source_id=admission_source_id,
        discharge_disposition_id=discharge_disposition_id,
        diag_1_grp=diag_1_grp,
        diag_2_grp=diag_2_grp,
        diag_3_grp=diag_3_grp,
        medical_specialty=medical_specialty,
        payer_code=payer_code,
        A1Cresult=A1Cresult,
        max_glu_serum=max_glu_serum,
        **{
            "drug_" + drug: value
            for drug, value in drug_vals.items()
        },
    )

    try:

        with st.spinner("Analyse du profil en cours..."):
            result = predict_patient(patient_input)

        X = result["X"]
        prediction = result["prediction"]
        probability = result["probability"]
        segment_id = result["segment_id"]
        segment_name = result["segment_name"]
        segment_description = result["segment_description"]

        # ---------------- RESULTS HEADER ----------------

        html(
            """
            <div class="results-header">
                <div class="eyebrow">Analyse terminée</div>
                <div class="results-title">Résultats du profil patient</div>
            </div>
            """
        )

        # ---------------- RESULT CARDS ----------------

        r1, r2, r3 = st.columns(3)

        with r1:
            html(
                f"""
                <div class="result-card">
                    <div class="result-label">Probabilité estimée</div>
                    <div class="result-value">{probability:.1%}</div>
                    <div class="result-small">Réadmission dans les 30 jours</div>
                </div>
                """
            )

        with r2:

            if prediction:
                decision = "Risque élevé"
                icon = ""
            else:
                decision = "Risque faible"
                icon = ""

            html(
                f"""
                <div class="result-card">
                    <div class="result-label">Décision du modèle</div>
                    <div class="result-value">{icon} {decision}</div>
                    <div class="result-small">Classification Linear SVM</div>
                </div>
                """
            )

        with r3:
            html(
                f"""
                <div class="result-card">
                    <div class="result-label">Profil patient</div>
                    <div class="result-value">Segment {segment_id}</div>
                    <div class="result-small">{segment_name}</div>
                </div>
                """
            )

        # ---------------- PROBABILITY BAR ----------------

        st.markdown("<br>", unsafe_allow_html=True)

        html(
            f"""
            <div style="display:flex; justify-content:space-between; color:#7D7067; font-size:0.76rem; font-weight:600; margin-bottom:0.4rem;">
                <span>Niveau de probabilité</span>
                <span>{probability:.1%}</span>
            </div>
            """
        )

        st.progress(min(probability / 0.50, 1.0))

        st.caption(
            "Échelle visuelle normalisée sur une plage de 0 à 50 %."
        )

        # ---------------- RISK MESSAGE ----------------

        if prediction:

            html(
                """
                <div class="risk-high">
                    <div class="risk-title">
                         Le modèle signale un profil à surveiller
                    </div>
                    <div class="risk-text">
                        Le SVM classe ce patient comme susceptible
                        d'être réadmis dans les 30 jours.
                        Cette alerte doit être interprétée avec
                        le contexte clinique réel.
                    </div>
                </div>
                """
            )

        else:

            html(
                """
                <div class="risk-low">
                    <div class="risk-title">
                        ✓ Aucun signal élevé détecté
                    </div>
                    <div class="risk-text">
                        Le SVM ne classe pas ce profil comme
                        présentant un risque élevé selon les
                        caractéristiques fournies.
                    </div>
                </div>
                """
            )

        # ---------------- SEGMENT INFORMATION ----------------

        html(
            f"""
            <div class="segment-card">
                <div class="segment-number">
                    Profil comportemental · Segment {segment_id}
                </div>
                <div class="segment-name">{segment_name}</div>
                <div class="segment-description">{segment_description}</div>
            </div>
            """
        )

        # ---------------- TECHNICAL DETAILS ----------------

        st.markdown("<br>", unsafe_allow_html=True)

        with st.expander("Voir les détails techniques de l'analyse"):

            tab1, tab2 = st.tabs(["Variables modèle", "Résumé patient"])

            with tab1:

                st.caption(
                    f"{len(X.columns)} variables envoyées au modèle."
                )

                st.dataframe(
                    X.T.rename(columns={0: "Valeur"}),
                    use_container_width=True,
                    height=450,
                )

            with tab2:

                summary = pd.DataFrame(
                    {
                        "Variable": [
                            "Âge",
                            "Sexe",
                            "Durée du séjour",
                            "Nombre de diagnostics",
                            "Analyses laboratoire",
                            "Procédures",
                            "Médicaments",
                            "Consultations externes",
                            "Passages aux urgences",
                            "Hospitalisations",
                            "Probabilité",
                            "Décision",
                            "Segment",
                        ],
                        "Valeur": [
                            f"{age} ans",
                            gender,
                            f"{time_in_hospital} jours",
                            str(number_diagnoses),
                            str(num_lab_procedures),
                            str(num_procedures),
                            str(num_medications),
                            str(number_outpatient),
                            str(number_emergency),
                            str(number_inpatient),
                            f"{probability:.1%}",
                            decision,
                            f"{segment_id} — {segment_name}",
                        ],
                    }
                )

                st.dataframe(
                    summary,
                    use_container_width=True,
                    hide_index=True,
                )

    except Exception as e:

        st.error("Une erreur est survenue pendant l'analyse.")

        with st.expander("Détails techniques"):
            st.code(repr(e), language="text")


# ================================================================
# FOOTER
# ================================================================

html(
    """
    <div class="disclaimer">
        <strong>Information importante :</strong>
        cette application est un démonstrateur d'intelligence
        artificielle destiné à l'analyse prédictive et à
        l'apprentissage.
        Les résultats ne constituent pas un diagnostic médical
        et ne doivent pas remplacer l'évaluation d'un professionnel
        de santé.
    </div>
    """
)