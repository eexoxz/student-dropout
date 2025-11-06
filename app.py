import os
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)

# ------------------ CONFIG ------------------
DATA_CANDIDATES = [
    "data/testing_cleaned.csv",
    "notebooks/testing_cleaned.csv",
    "testing_cleaned.csv",
]
MODEL_CANDIDATES = [
    "models/random_forest_pipeline.joblib",
    "notebooks/random_forest_pipeline.joblib",
    "random_forest_pipeline.joblib",
]

NUMERIC = [
    "Hours_Studied","Attendance","Sleep_Hours","Previous_Scores",
    "Tutoring_Sessions","Physical_Activity"
]
CATEGORICAL = [
    "Parental_Involvement","Access_to_Resources","Extracurricular_Activities",
    "Motivation_Level","Internet_Access","Family_Income","Teacher_Quality",
    "School_Type","Peer_Influence","Learning_Disabilities",
    "Parental_Education_Level","Distance_from_Home","Gender"
]
CHOICES = {
    "Parental_Involvement": ["Low","Medium","High"],
    "Access_to_Resources": ["Low","Medium","High"],
    "Extracurricular_Activities": ["No","Yes"],
    "Motivation_Level": ["Low","Medium","High"],
    "Internet_Access": ["No","Yes"],
    "Family_Income": ["Low","Medium","High"],
    "Teacher_Quality": ["Low","Medium","High"],
    "School_Type": ["Public","Private"],
    "Peer_Influence": ["Negative","Neutral","Positive"],
    "Learning_Disabilities": ["No","Yes"],
    "Parental_Education_Level": ["Primary","Secondary","Diploma","High School","Bachelor","Master","PhD"],
    "Distance_from_Home": ["Near","Moderate","Far"],
    "Gender": ["Male","Female"],
}
TARGET = "Performance"

# ------------------ HELPERS ------------------
def first_existing(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"None of these paths exist: {paths}")

def clean_text_df(df: pd.DataFrame) -> pd.DataFrame:
    df2 = df.copy()
    for c in df2.select_dtypes(include="object").columns:
        df2[c] = df2[c].astype(str).str.strip().str.title()
    return df2

@st.cache_resource
def load_model():
    path = first_existing(MODEL_CANDIDATES)
    return joblib.load(path), path

@st.cache_data
def load_data():
    path = first_existing(DATA_CANDIDATES)
    df = pd.read_csv(path)
    return df, path

def ensure_target(df: pd.DataFrame) -> pd.DataFrame:
    if TARGET in df.columns:
        return df
    # fallback: derive from Exam_Score tertiles
    if "Exam_Score" in df.columns:
        df = df.copy()
        df["Exam_Score"] = pd.to_numeric(df["Exam_Score"], errors="coerce")
        df = df.dropna(subset=["Exam_Score"])
        df[TARGET] = pd.qcut(
            df["Exam_Score"].rank(method="first"),
            q=[0, 1/3, 2/3, 1.0],
            labels=["Low","Medium","High"]
        )
        st.info("Target 'Performance' was missing. Created from Exam_Score (tertiles).")
        return df
    st.error("Neither 'Performance' nor 'Exam_Score' found. Cannot proceed.")
    st.stop()

def feature_importances(pipe):
    try:
        pre = pipe.named_steps["pre"]          # ColumnTransformer
        clf = pipe.named_steps["clf"]          # RandomForestClassifier expected
        num_names = pre.transformers_[0][2]
        ohe = pre.named_transformers_["cat"].named_steps["encoder"]
        cat_names = list(ohe.get_feature_names_out(pre.transformers_[1][2]))
        all_names = list(num_names) + cat_names
        if hasattr(clf, "feature_importances_"):
            imp = pd.DataFrame({"Feature": all_names, "Importance": clf.feature_importances_})
            return imp.sort_values("Importance", ascending=False)
    except Exception as e:
        st.warning(f"Could not compute feature importances: {e}")
    return None

# ------------------ APP ------------------
st.set_page_config(page_title="Student Analytics Dashboard", layout="wide")
st.title("🎓 Student Analytics Dashboard")

pipe, model_path = load_model()
df_raw, data_path = load_data()
st.caption(f"Model: `{model_path}`  |  Data: `{data_path}`")

# clean + ensure target BEFORE anything else
df = clean_text_df(df_raw)
df = ensure_target(df)

tab_overview, tab_predict, tab_perf = st.tabs(["📊 Data Overview", "🔮 Predict", "🧪 Model Performance"])

# ------------------ TAB 1: DATA OVERVIEW ------------------
with tab_overview:
    st.subheader("Attributes Distribution")

    # Numeric histograms
    num_cols = [c for c in NUMERIC if c in df.columns]
    if num_cols:
        st.markdown("**Numeric Feature Distribution**")
        cols = st.columns(min(4, len(num_cols)))
        for i, col in enumerate(num_cols):
            fig = px.histogram(df, x=col, nbins=30, title=col)
            cols[i % len(cols)].plotly_chart(fig, use_container_width=True)

    # Categorical bars
    cat_cols = [c for c in CATEGORICAL if c in df.columns]
    if cat_cols:
        st.markdown("**Categorical Feature Distribution**")
        cols = st.columns(min(4, len(cat_cols)))
        for i, col in enumerate(cat_cols):
            vc = df[col].value_counts(dropna=False).rename_axis(col).reset_index(name="Count")
            fig = px.bar(vc, x=col, y="Count", title=f"{col} Count")
            cols[i % len(cols)].plotly_chart(fig, use_container_width=True)

    # Target distribution (counts/%)
    st.markdown("**Target Distribution**")
    vc = df[TARGET].value_counts(dropna=False).rename_axis(TARGET).reset_index(name="Count")
    vc["Percent"] = (vc["Count"] / vc["Count"].sum() * 100).round(2)
    view = st.radio("Show:", ["Counts", "Percent"], horizontal=True, key="target_view")
    if view == "Counts":
        fig = px.bar(vc, x=TARGET, y="Count", text="Count", title="Target Distribution (Counts)")
        fig.update_traces(textposition="outside")
        fig.update_yaxes(range=[0, vc["Count"].max() * 1.15])
    else:
        fig = px.bar(vc, x=TARGET, y="Percent", text="Percent", title="Target Distribution (%)")
        fig.update_traces(textposition="outside")
        fig.update_yaxes(range=[0, 100])
    st.plotly_chart(fig, use_container_width=True)

# ------------------ TAB 2: PREDICT ------------------
with tab_predict:
    st.subheader("Single Prediction")

    # quick presets (helps debug "always Low")
    preset = st.radio("Preset:", ["Custom", "Likely High", "Likely Low"], horizontal=True)
    rec = {}

    if preset == "Likely High":
        rec = {
            "Hours_Studied": 8, "Attendance": 96, "Sleep_Hours": 8,
            "Previous_Scores": 85, "Tutoring_Sessions": 2, "Physical_Activity": 4,
            "Parental_Involvement": "High", "Access_to_Resources": "High",
            "Extracurricular_Activities": "Yes", "Motivation_Level": "High",
            "Internet_Access": "Yes", "Family_Income": "High", "Teacher_Quality": "High",
            "School_Type": "Public", "Peer_Influence": "Positive",
            "Learning_Disabilities": "No", "Parental_Education_Level": "Bachelor",
            "Distance_from_Home": "Near", "Gender": "Male"
        }
    elif preset == "Likely Low":
        rec = {
            "Hours_Studied": 1, "Attendance": 70, "Sleep_Hours": 5,
            "Previous_Scores": 55, "Tutoring_Sessions": 0, "Physical_Activity": 1,
            "Parental_Involvement": "Low", "Access_to_Resources": "Low",
            "Extracurricular_Activities": "No", "Motivation_Level": "Low",
            "Internet_Access": "No", "Family_Income": "Low", "Teacher_Quality": "Low",
            "School_Type": "Public", "Peer_Influence": "Negative",
            "Learning_Disabilities": "Yes", "Parental_Education_Level": "Primary",
            "Distance_from_Home": "Far", "Gender": "Female"
        }

    c1, c2, c3 = st.columns(3)
    # numeric
    for k, default, col in [
        ("Hours_Studied", 6.0, c1),
        ("Attendance", 92.0, c2),
        ("Sleep_Hours", 7.0, c3),
        ("Previous_Scores", 70.0, c1),
        ("Tutoring_Sessions", 1.0, c2),
        ("Physical_Activity", 3.0, c3),
    ]:
        rec[k] = col.number_input(k, value=float(rec.get(k, default)))

    # categorical
    cols = st.columns(3)
    for i, col in enumerate(CATEGORICAL):
        opts = CHOICES.get(col, sorted(df[col].dropna().unique().tolist()))
        rec[col] = cols[i % 3].selectbox(col, options=opts, index=opts.index(rec[col]) if col in rec and rec[col] in opts else 0)

    if st.button("Predict", type="primary"):
        X_one = pd.DataFrame([rec])
        X_one = clean_text_df(X_one)
        pred = pipe.predict(X_one)[0]

        # show probabilities to diagnose bias
        proba_txt = ""
        try:
            probs = pipe.predict_proba(X_one)[0]
            classes = list(pipe.named_steps["clf"].classes_)
            proba_df = pd.DataFrame({"Class": classes, "Prob (%)": probs * 100}).sort_values("Prob (%)", ascending=False)
            proba_txt = proba_df.to_string(index=False)
        except Exception:
            pass

        st.success(f"Prediction: **{pred}**")
        if proba_txt:
            st.text("Class probabilities (%):")
            st.text(proba_txt)

    st.divider()
    st.subheader("Batch Prediction (CSV)")
    st.caption("Upload a CSV with only the model feature columns.")
    up = st.file_uploader("Upload CSV", type=["csv"])
    if up is not None:
        df_in = pd.read_csv(up)
        df_in = clean_text_df(df_in)
        preds = pipe.predict(df_in)
        out = df_in.copy()
        out["Prediction"] = preds
        st.dataframe(out.head(50))
        st.download_button("Download predictions", out.to_csv(index=False), file_name="predictions.csv", type="primary")

# ------------------ TAB 3: PERFORMANCE ------------------
with tab_perf:
    st.subheader("Model Performance on Full Dataset")

    drop_cols = [c for c in ["Performance", "Exam_Score"] if c in df.columns]
    X = df.drop(columns=drop_cols)
    y = df[TARGET]

    y_pred = pipe.predict(X)

    acc = accuracy_score(y, y_pred)
    st.metric("Accuracy", f"{acc*100:.2f}%")

    labels = sorted(pd.unique(y))
    prec, rec, f1, support = precision_recall_fscore_support(y, y_pred, labels=labels, zero_division=0)
    scores_df = pd.DataFrame({
        "Class": labels,
        "Precision (%)": prec * 100,
        "Recall (%)": rec * 100,
        "F1 (%)": f1 * 100,
        "Support": support
    }).sort_values("Class")
    st.markdown("**Per-Class Metrics**")
    st.dataframe(scores_df.style.format({"Precision (%)":"{:.2f}","Recall (%)":"{:.2f}","F1 (%)":"{:.2f}"}))

    cm = confusion_matrix(y, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=[f"True {c}" for c in labels], columns=[f"Pred {c}" for c in labels])
    st.markdown("**Confusion Matrix**")
    st.dataframe(cm_df)

    st.markdown("**Top Feature Importances**")
    imp_df = feature_importances(pipe)
    if imp_df is not None:
        st.dataframe(imp_df.head(20))
        fig = px.bar(imp_df.head(20).sort_values("Importance"),
                     x="Importance", y="Feature", orientation="h",
                     title="Top 20 Features")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Classifier does not expose feature importances.")
