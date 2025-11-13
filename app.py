import pandas as pd
import numpy as np
import joblib
import streamlit as st
import altair as alt
import plotly.express as px
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ===== FIXED CONFIG (no UI) =====
MODEL_PATH   = "models/logistic_regression_student.joblib"
CSV_PATH     = "student_performance_cleaned.csv"
SCORE_COL    = "Exam_Score"
HI_THRESHOLD = 70
TARGET_NAME  = "Performance"
TEST_SIZE    = 0.20
RANDOM_STATE = 42

# ===== App setup =====
st.set_page_config(page_title="Student Risk Dashboard", page_icon="🎓", layout="wide")
st.title("Student Risk Dashboard")

@st.cache_data
def load_csv(path_or_file):
    return pd.read_csv(path_or_file)

def ensure_binary_target(df, score_col=SCORE_COL, threshold=HI_THRESHOLD, target_name=TARGET_NAME):
    if target_name not in df.columns:
        df[target_name] = (df[score_col] >= threshold).astype(int)
    return df

@st.cache_resource
def load_model(path):
    if not Path(path).exists():
        st.error(f"Model not found: {path}")
        st.stop()
    return joblib.load(path)

# ===== Load data & model (no UI controls) =====
df = load_csv(CSV_PATH)
df = ensure_binary_target(df.copy())

drop_cols = [TARGET_NAME]
if SCORE_COL in df.columns:
    drop_cols.append(SCORE_COL)
feature_cols = [c for c in df.columns if c not in drop_cols]

CANDIDATES = {
    "Hours_Studied":         ["Hours_Studied", "Hours_Stu", "Hours"],
    "Attendance":            ["Attendance"],
    "Parental_Involvement":  ["Parental_Involvement", "Parental_I"],
    "Access_to_Resources":   ["Access_to_Resources", "Access_to"],
    "Extracurricular":       ["Extracurricular", "Extracurric", "Extracurricular_Activities"],
    "Sleep_Hours":           ["Sleep_Hours", "Sleep_Hou"],
    "Previous_Scores":       ["Previous_Scores", "Previous_S"],
    "Motivation":            ["Motivation", "Motivation_Level"],
    "Internet_Access":       ["Internet_Access", "Internet_A"],
    "Tutoring_Sessions":     ["Tutoring_Sessions", "Tutoring_S"],
    "Family_Income":         ["Family_Income", "Family_Inc"],
    "Teacher_Quality":       ["Teacher_Quality", "Teacher_Q"],
    "School_Type":           ["School_Type", "School_Typ"],
    "Peer_Influence":        ["Peer_Influence", "Peer_Influe", "Peer_Influ"],
    "Physical_Activity":     ["Physical_Activity", "Physical_A"],
    "Learning_Disabilities": ["Learning_Disabilities", "Learning_D"],
    "Parental_Education":    ["Parental_Education", "Parental_E", "Parental_Education_Level"],
    "Distance_from_School":  ["Distance_from_School", "Distance_f", "Distance_from_Home"],
    "Gender":                ["Gender"],
    "Exam_Score":            ["Exam_Score"]
}

def resolve_columns(df_cols, candidates):
    res = {}
    for canon, opts in candidates.items():
        hit = None
        for o in opts:
            m = next((c for c in df_cols if c == o or c.startswith(o)), None)
            if m:
                hit = m
                break
        if hit:
            res[canon] = hit
    return res

COL = resolve_columns(df.columns.tolist(), CANDIDATES)

X = df[feature_cols]
y = df[TARGET_NAME]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)

model = load_model(MODEL_PATH)

AT_RISK_MAX = 0.40
VERY_SAFE_MIN = 0.70
DECISION_THRESHOLD = 0.50

st.markdown("---")

# ===== LEFT NAV + RIGHT CONTENT =====
nav_col, main_col = st.columns([1, 4])

with nav_col:
    st.markdown("## 📋 Menu")
    page = st.radio(
        "Navigation",
        ["Analysis", "Risk Tool", "Model Performance"],
        label_visibility="collapsed"
    )

with main_col:

    # ---------- PAGE 1: ANALYSIS ----------
    if page == "Analysis":
        st.markdown("## 📊 Analysis")

        # --- Key driver graphs ---
        st.subheader("Key driver graphs (binned value → avg predicted safety %)")

        probs_all = model.predict_proba(X)[:, 1]
        df_pred = X.copy()
        df_pred["Prob_Safe_%"] = probs_all * 100

        def binned_bar(df_in, feature, bins=10):
            if feature not in df_in.columns:
                st.info(f"Feature `{feature}` not found.")
                return
            cuts = pd.cut(df_in[feature], bins=bins, include_lowest=True)
            g = df_in.groupby(cuts, observed=False)["Prob_Safe_%"].mean().reset_index()
            g["Bin"] = g[feature].astype(str)
            fig = px.bar(
                g, x="Bin", y="Prob_Safe_%",
                labels={"Prob_Safe_%": "Avg Safety (%)", "Bin": "Value bin"},
                title=feature
            )
            fig.update_layout(margin=dict(l=0, r=0, t=50, b=0), height=300)
            st.plotly_chart(fig, use_container_width=True)

        key_feats = [COL.get("Hours_Studied"), COL.get("Attendance"),
                     COL.get("Sleep_Hours"), COL.get("Previous_Scores")]
        key_feats = [k for k in key_feats if k is not None]

        g1, g2 = st.columns(2)
        if key_feats:
            with g1:
                binned_bar(df_pred, key_feats[0])
            with g2:
                if len(key_feats) > 1:
                    binned_bar(df_pred, key_feats[1])
            g3, g4 = st.columns(2)
            with g3:
                if len(key_feats) > 2:
                    binned_bar(df_pred, key_feats[2])
            with g4:
                if len(key_feats) > 3:
                    binned_bar(df_pred, key_feats[3])
        else:
            st.info("Key driver columns were not detected in your file.")

        st.divider()

        # --- Dataset overview ---
        c1, c2, c3 = st.columns(3)
        c1.metric("Rows", f"{df.shape[0]:,}")
        c2.metric("Columns", f"{df.shape[1]:,}")
        c3.metric("Class 1 share", f"{(y.mean() * 100):.2f}%")

        st.write("**Preview**")
        st.dataframe(df.head())

        st.write("**Numeric summary**")
        st.dataframe(df.select_dtypes(include=[np.number]).describe().T)

        st.divider()
        st.write("### Histogram")
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if num_cols:
            sel = st.selectbox("Numeric column", num_cols, index=0)
            bins = st.slider("Bins", 5, 60, 20, 1)
            chart = (
                alt.Chart(df)
                .mark_bar()
                .encode(
                    x=alt.X(f"{sel}:Q", bin=alt.Bin(maxbins=bins), title=sel),
                    y=alt.Y("count():Q", title="Frequency"),
                    tooltip=[
                        alt.Tooltip(f"{sel}:Q", bin=alt.Bin(maxbins=bins)),
                        alt.Tooltip("count():Q"),
                    ],
                )
                .properties(height=300)
            )
            st.altair_chart(chart, use_container_width=True)

        st.divider()
        st.write("### 🔥 Correlation Heatmap (numeric features)")
        corr = df.select_dtypes(include=[np.number]).corr(numeric_only=True)
        fig = px.imshow(
            corr,
            color_continuous_scale="RdBu",
            zmin=-1,
            zmax=1,
            aspect="auto",
            title="Correlation Heatmap",
        )
        fig.update_layout(margin=dict(l=0, r=0, t=40, b=0), height=520)
        st.plotly_chart(fig, use_container_width=True)

    # ---------- PAGE 2: RISK TOOL ----------
    elif page == "Risk Tool":
        st.markdown("## 🧪 Risk Tool – Predict a Student")

        BIN = {"No": 0, "Yes": 1}
        GENDER = {"Female": 0, "Male": 1}
        L_M_H = {"Low": 0.0, "Medium": 0.5, "High": 1.0}
        PEER = {"Negative": 0.0, "Neutral": 0.5, "Positive": 1.0}
        SCHOOL = {"Private": 0, "Public": 1}
        DIST = {"Far": 0.0, "Moderate": 0.5, "Near": 1.0}
        P_EDU = {"High School": 0.0, "College": 0.5, "Postgraduate": 1.0}
        PHYS = {"None": 0.0, "Low": 1 / 3, "Medium": 2 / 3, "High": 1.0}

        c1, c2 = st.columns(2)
        ui = {}
        ui[COL["Hours_Studied"]] = c1.slider(
            "Hours_Studied (0–1)", 0.0, 1.0, float(df[COL["Hours_Studied"]].median()), 0.01
        )
        ui[COL["Attendance"]] = c2.slider(
            "Attendance (0–1)", 0.0, 1.0, float(df[COL["Attendance"]].median()), 0.01
        )
        ui[COL["Sleep_Hours"]] = c1.slider(
            "Sleep_Hours (0–1)", 0.0, 1.0, float(df[COL["Sleep_Hours"]].median()), 0.01
        )
        ui[COL["Previous_Scores"]] = c2.slider(
            "Previous_Scores (0–1)",
            0.0,
            1.0,
            float(df[COL["Previous_Scores"]].median()),
            0.01,
        )

        ui[COL["Parental_Involvement"]] = L_M_H[
            c1.selectbox("Parental_Involvement", list(L_M_H.keys()), index=1)
        ]
        ui[COL["Access_to_Resources"]] = L_M_H[
            c2.selectbox("Access_to_Resources", list(L_M_H.keys()), index=1)
        ]
        ui[COL["Extracurricular"]] = BIN[
            c1.selectbox("Extracurricular", list(BIN.keys()), index=0)
        ]
        ui[COL["Motivation"]] = L_M_H[
            c2.selectbox("Motivation", list(L_M_H.keys()), index=1)
        ]
        ui[COL["Internet_Access"]] = BIN[
            c1.selectbox("Internet_Access", list(BIN.keys()), index=1)
        ]
        ui[COL["Tutoring_Sessions"]] = c2.select_slider(
            "Tutoring_Sessions (scaled)",
            options=[0.0, 0.25, 0.5, 0.75, 1.0],
            value=0.0,
        )
        ui[COL["Family_Income"]] = L_M_H[
            c1.selectbox("Family_Income", list(L_M_H.keys()), index=1)
        ]
        ui[COL["Teacher_Quality"]] = L_M_H[
            c2.selectbox("Teacher_Quality", list(L_M_H.keys()), index=1)
        ]
        ui[COL["School_Type"]] = SCHOOL[
            c1.selectbox("School_Type", list(SCHOOL.keys()), index=1)
        ]
        ui[COL["Peer_Influence"]] = PEER[
            c2.selectbox("Peer_Influence", list(PEER.keys()), index=2)
        ]
        ui[COL["Physical_Activity"]] = PHYS[
            c1.selectbox("Physical_Activity", list(PHYS.keys()), index=2)
        ]
        ui[COL["Learning_Disabilities"]] = BIN[
            c2.selectbox("Learning_Disabilities", list(BIN.keys()), index=0)
        ]
        ui[COL["Parental_Education"]] = P_EDU[
            c1.selectbox("Parental_Education", list(P_EDU.keys()), index=1)
        ]
        ui[COL["Distance_from_School"]] = DIST[
            c2.selectbox("Distance_from_School", list(DIST.keys()), index=2)
        ]
        ui[COL["Gender"]] = GENDER[
            c1.selectbox("Gender", list(GENDER.keys()), index=1)
        ]

        for col in feature_cols:
            if col not in ui:
                ui[col] = (
                    float(df[col].median())
                    if pd.api.types.is_numeric_dtype(df[col])
                    else 0.0
                )

        X_user = pd.DataFrame([ui])[feature_cols]
        prob_safe = float(model.predict_proba(X_user)[:, 1][0])

        band = (
            "Very Safe"
            if prob_safe >= VERY_SAFE_MIN
            else ("At-Risk" if prob_safe <= AT_RISK_MAX else "Neutral")
        )

        st.divider()
        st.metric("Predicted probability (Safe / Class 1)", f"{prob_safe * 100:.2f}%")
        st.metric("Risk Band", band)
        st.progress(min(max(prob_safe, 0), 1))

    # ---------- PAGE 3: MODEL PERFORMANCE ----------
    else:  # "Model Performance"
        st.markdown("## 📈 Model Performance")

        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= DECISION_THRESHOLD).astype(int)
        acc = accuracy_score(y_test, y_pred)
        st.metric("Logistic Regression Accuracy", f"{acc * 100:.2f}%")
        st.caption(
            f"Hold-out size: {len(y_test)}; decision threshold: {DECISION_THRESHOLD:.2f}; "
            f"test size: {TEST_SIZE:.2f}; random state: {RANDOM_STATE}"
        )

        st.divider()
        st.subheader("Feature importance (all |logistic coefficients|)")
        inner = model[-1] if hasattr(model, "__getitem__") else model
        try:
            coefs = inner.coef_[0]
            imp_all = (
                pd.DataFrame(
                    {
                        "Feature": feature_cols,
                        "Coefficient": coefs,
                        "|Coefficient|": np.abs(coefs),
                        "Odds Ratio (per +0.1)": np.exp(coefs * 0.1),
                    }
                )
                .sort_values("|Coefficient|", ascending=False)
            )

            fig = px.bar(
                imp_all.iloc[::-1],
                x="Coefficient",
                y="Feature",
                hover_data={
                    "Odds Ratio (per +0.1)": ":.3f",
                    "Coefficient": ":.3f",
                    "Feature": True,
                    "|Coefficient|": ":.3f",
                },
                title="All features (+: increases odds of Safe; −: decreases)",
            )
            st.plotly_chart(fig, use_container_width=True)

            st.write("**Full table**")
            st.dataframe(imp_all.reset_index(drop=True))
        except Exception as e:
            st.warning(f"Could not read coefficients: {e}")

        st.divider()
        st.subheader("Students by risk band (full dataset)")
        probs_all = model.predict_proba(X)[:, 1]
        out = pd.DataFrame(
            {
                "Student_ID": [f"{i:04d}" for i in range(1, len(df) + 1)],
                "Prob_Safe (%)": (probs_all * 100).round(2),
            }
        )
        at_risk = (
            out[out["Prob_Safe (%)"] <= AT_RISK_MAX * 100]
            .copy()
            .sort_values("Prob_Safe (%)")
        )
        neutral = (
            out[
                (out["Prob_Safe (%)"] > AT_RISK_MAX * 100)
                & (out["Prob_Safe (%)"] < VERY_SAFE_MIN * 100)
            ]
            .copy()
            .sort_values("Prob_Safe (%)", ascending=False)
        )
        very_safe = (
            out[out["Prob_Safe (%)"] >= VERY_SAFE_MIN * 100]
            .copy()
            .sort_values("Prob_Safe (%)", ascending=False)
        )

        c1, c2, c3 = st.columns(3)
        c1.metric("At-Risk (≤ 40%)", f"{len(at_risk):,}")
        c2.metric("Neutral (40–70%)", f"{len(neutral):,}")
        c3.metric("Very Safe (≥ 70%)", f"{len(very_safe):,}")

        st.write("**At-Risk (lowest → highest)**")
        st.dataframe(at_risk.reset_index(drop=True))
        st.write("**Very Safe (highest → lowest)**")
        st.dataframe(very_safe.reset_index(drop=True))

        st.download_button(
            "Download At-Risk CSV",
            at_risk.to_csv(index=False).encode("utf-8"),
            "at_risk_students.csv",
            "text/csv",
        )
        st.download_button(
            "Download Very Safe CSV",
            very_safe.to_csv(index=False).encode("utf-8"),
            "very_safe_students.csv",
            "text/csv",
        )
