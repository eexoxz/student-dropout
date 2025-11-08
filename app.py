import pandas as pd
import numpy as np
import joblib
import streamlit as st

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, classification_report,
    confusion_matrix, roc_curve
)
import matplotlib.pyplot as plt

st.set_page_config(page_title="Student Risk Dashboard", layout="wide")
st.title("🎓 Student Performance & Dropout Risk Dashboard")

@st.cache_data
def load_csv(path_or_file):
    return pd.read_csv(path_or_file)

def pct(x): 
    return f"{x*100:.2f}%"

def ensure_binary_target(df, score_col="Exam_Score", threshold=70, target_name="Performance"):
    # create 1=high achiever / safe, 0=risk — adjust names if you want inverted
    if target_name not in df.columns:
        df[target_name] = (df[score_col] >= threshold).astype(int)
    return df

def load_models():
    models = {}
    paths = {
        "Logistic Regression": "models/logistic_regression_student.joblib",
        "Decision Tree": "models/decision_tree_student.joblib",
        "Random Forest": "models/random_forest_student.joblib",
    }
    for name, p in paths.items():
        if Path(p).exists():
            models[name] = joblib.load(p)
    return models

def plot_confusion_matrix(y_true, y_pred, title, normalize=False):
    cm = confusion_matrix(y_true, y_pred, normalize=("true" if normalize else None))
    labels = ["Class 0", "Class 1"]
    df_cm = pd.DataFrame(cm, index=labels, columns=labels)
    plt.figure(figsize=(4.2, 3.6))
    fmt = ".2%" if normalize else "d"
    import seaborn as sns
    sns.heatmap(df_cm, annot=True, fmt=fmt, cbar=False, linewidths=.5)
    plt.title(title + (" — %" if normalize else " — counts"))
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    st.pyplot(plt.gcf())
    plt.close()

def plot_roc(y_true, y_prob, title):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    plt.figure(figsize=(4.2, 3.4))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_true, y_prob):.3f}")
    plt.plot([0,1], [0,1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend()
    st.pyplot(plt.gcf())
    plt.close()

st.sidebar.header("⚙️ Settings")

uploaded = st.sidebar.file_uploader("Upload your numeric CSV", type=["csv"])
default_path = st.sidebar.text_input("…or type CSV path", "student_performance_cleaned.csv")

data_src = uploaded if uploaded is not None else default_path
df = load_csv(data_src)

# adjust these to match your file
score_col = st.sidebar.text_input("Score column (for target creation)", "Exam_Score")
threshold = st.sidebar.number_input("High achiever threshold", 0, 100, 70, step=1)
target_name = st.sidebar.text_input("Binary target name", "Performance")

# create target if absent
df = ensure_binary_target(df.copy(), score_col=score_col, threshold=threshold, target_name=target_name)

# features used for models (drop target and score)
drop_cols = [target_name]
if score_col in df.columns: drop_cols.append(score_col)
feature_cols = [c for c in df.columns if c not in drop_cols]
# ---- COLUMN RESOLVER (maps your short/long column names) ----
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
            matches = [c for c in df_cols if c == o or c.startswith(o)]
            if matches:
                hit = matches[0]; break
        if hit: res[canon] = hit
    return res

COL = resolve_columns(df.columns.tolist(), CANDIDATES)
# (optional debug) show mapping in sidebar:
# st.sidebar.json(COL)

X = df[feature_cols]
y = df[target_name]

# split for evaluation inside the app (keeps it consistent)
test_size = st.sidebar.slider("Test size", 0.1, 0.4, 0.2, 0.05)
random_state = st.sidebar.number_input("Random state", 0, 9999, 42, step=1)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=random_state, stratify=y
)

models = load_models()
model_name = st.sidebar.selectbox("Model for Risk Tool", list(models.keys()) or ["(no models found)"])
threshold_prob = st.sidebar.slider("Risk cut-off (probability)", 0.05, 0.95, 0.5, 0.01)

tab1, tab2, tab3 = st.tabs(["📊 Dataset Analysis", "🧪 Risk Tool", "📈 Model Performance"])

with tab1:
    st.subheader("Dataset Overview")
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", f"{df.shape[0]:,}")
    c2.metric("Columns", f"{df.shape[1]:,}")
    c3.metric("Target balance (Class 1)", pct(y.mean()))

    st.write("**Preview**")
    st.dataframe(df.head())

    st.write("**Summary (Numeric)**")
    st.dataframe(df.select_dtypes(include=[np.number]).describe())

    st.write("**Histogram (pick a column)**")
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if num_cols:
        sel = st.selectbox("Numeric column", num_cols, index=0)
        st.bar_chart(df[sel].value_counts().sort_index() if df[sel].nunique()<50 else df[sel])

    st.write("**Correlation Heatmap**")
    import seaborn as sns
    corr = df.select_dtypes(include=[np.number]).corr(numeric_only=True)
    plt.figure(figsize=(8,6))
    sns.heatmap(corr, cmap="coolwarm", center=0)
    st.pyplot(plt.gcf())
    plt.close()

# --- TAB 2: Risk Tool (DISCRETE inputs tied to your encodings) ---
with tab2:
    st.subheader("Interactive Risk Prediction (discrete inputs)")

    if not models:
        st.warning("No saved models found in `models/`.")
    else:
        model = models[model_name]

        # mappings (adjust if your encodings differ)
        BIN    = {"No": 0, "Yes": 1}
        GENDER = {"Female": 0, "Male": 1}
        L_M_H  = {"Low": 0.0, "Medium": 0.5, "High": 1.0}
        PEER   = {"Negative": 0.0, "Neutral": 0.5, "Positive": 1.0}
        SCHOOL = {"Private": 0, "Public": 1}
        DIST   = {"Far": 0.0, "Moderate": 0.5, "Near": 1.0}
        P_EDU  = {"High School": 0.0, "College": 0.5, "Postgraduate": 1.0}
        PHYS   = {"None": 0.0, "Low": 1/3, "Medium": 2/3, "High": 1.0}

        c1, c2 = st.columns(2)
        ui = {}

        # continuous (already scaled 0–1)
        ui[COL["Hours_Studied"]]   = c1.number_input("Hours_Studied (0–1)", 0.0, 1.0, float(df[COL["Hours_Studied"]].median()), 0.01)
        ui[COL["Attendance"]]      = c2.number_input("Attendance (0–1)", 0.0, 1.0, float(df[COL["Attendance"]].median()), 0.01)
        ui[COL["Sleep_Hours"]]     = c1.number_input("Sleep_Hours (0–1)", 0.0, 1.0, float(df[COL["Sleep_Hours"]].median()), 0.01)
        ui[COL["Previous_Scores"]] = c2.number_input("Previous_Scores (0–1)", 0.0, 1.0, float(df[COL["Previous_Scores"]].median()), 0.01)

        # discrete categoricals (fixed encoded values)
        ui[COL["Parental_Involvement"]]  = L_M_H[c1.selectbox("Parental_Involvement", list(L_M_H.keys()), index=1)]
        ui[COL["Access_to_Resources"]]   = L_M_H[c2.selectbox("Access_to_Resources",  list(L_M_H.keys()), index=1)]
        ui[COL["Extracurricular"]]       = BIN[c1.selectbox("Extracurricular", list(BIN.keys()), index=0)]
        ui[COL["Motivation"]]            = L_M_H[c2.selectbox("Motivation", list(L_M_H.keys()), index=1)]
        ui[COL["Internet_Access"]]       = BIN[c1.selectbox("Internet_Access", list(BIN.keys()), index=1)]
        ui[COL["Tutoring_Sessions"]]     = c2.select_slider("Tutoring_Sessions (scaled)", options=[0.0, 0.25, 0.5, 0.75, 1.0], value=0.0)
        ui[COL["Family_Income"]]         = L_M_H[c1.selectbox("Family_Income", list(L_M_H.keys()), index=1)]
        ui[COL["Teacher_Quality"]]       = L_M_H[c2.selectbox("Teacher_Quality", list(L_M_H.keys()), index=1)]
        ui[COL["School_Type"]]           = SCHOOL[c1.selectbox("School_Type", list(SCHOOL.keys()), index=1)]
        ui[COL["Peer_Influence"]]        = PEER[c2.selectbox("Peer_Influence", list(PEER.keys()), index=2)]
        ui[COL["Physical_Activity"]]     = PHYS[c1.selectbox("Physical_Activity", list(PHYS.keys()), index=2)]
        ui[COL["Learning_Disabilities"]] = BIN[c2.selectbox("Learning_Disabilities", list(BIN.keys()), index=0)]
        ui[COL["Parental_Education"]]    = P_EDU[c1.selectbox("Parental_Education", list(P_EDU.keys()), index=1)]
        ui[COL["Distance_from_School"]]  = DIST[c2.selectbox("Distance_from_School", list(DIST.keys()), index=2)]
        ui[COL["Gender"]]                = GENDER[c1.selectbox("Gender", list(GENDER.keys()), index=1)]

        X_user = pd.DataFrame([ui])[feature_cols]

        prob1 = float(model.predict_proba(X_user)[:, 1][0])
        label = int(prob1 >= threshold_prob)
        tag   = "Class 1 (High/Safe)" if label==1 else "Class 0 (At-Risk/Other)"

        st.divider()
        st.markdown(f"**Model:** {model_name}")
        st.metric("Predicted probability (Class 1)", f"{prob1*100:.2f}%")
        st.metric("Label", tag)
        st.progress(min(max(prob1,0),1))


# --- TAB 3: Model Performance (table summary + confusion matrices + feature importance) ---
with tab3:
    st.subheader("Evaluation on Hold-out Test Set")

    if not models:
        st.info("Load models in the sidebar to see performance.")
    else:
        from sklearn.model_selection import cross_val_score

        # -------------------- METRICS TABLE --------------------
        rows, cms = [], {}
        for name, m in models.items():
            y_pred = m.predict(X_test)
            y_prob = m.predict_proba(X_test)[:, 1] if hasattr(m, "predict_proba") else None

            cv_acc = cross_val_score(m, X_train, y_train, cv=5, scoring="accuracy").mean()
            acc    = accuracy_score(y_test, y_pred)
            f1m    = f1_score(y_test, y_pred, average="macro")
            auc    = roc_auc_score(y_test, y_prob) if y_prob is not None else np.nan

            rows.append([name, cv_acc, acc, f1m, auc])
            cms[name] = confusion_matrix(y_test, y_pred)

        perf = pd.DataFrame(rows, columns=["Model","CV Accuracy","Accuracy","Macro F1","AUC"]).set_index("Model")
        def pctcol(s): return (s*100).map(lambda v: f"{v:.2f}%")
        pretty = perf.copy()
        for c in ["CV Accuracy","Accuracy","Macro F1","AUC"]:
            pretty[c] = pctcol(pretty[c])

        st.dataframe(pretty)
        perf.to_csv("model_performance_summary.csv")
        pretty.to_csv("model_performance_summary_pretty.csv")
        st.caption("Saved: model_performance_summary.csv & model_performance_summary_pretty.csv")

        # -------------------- CONFUSION MATRICES --------------------
        st.divider()
        st.subheader("Confusion Matrices (counts)")
        for name, cm in cms.items():
            st.markdown(f"**{name}**")
            st.dataframe(pd.DataFrame(cm, index=["Actual 0","Actual 1"], columns=["Pred 0","Pred 1"]))

        # -------------------- FEATURE IMPORTANCE --------------------
        st.divider()
        st.subheader("Feature Importance")

        # Logistic Regression: show coef, OR per +1, and OR per +0.1 step (more interpretable)
        def show_logit_importance(model, features):
            import numpy as np
            coef = pd.DataFrame({
                "Feature": features,
                "Coefficient": model.coef_[0]
            })
            coef["|Coefficient|"] = coef["Coefficient"].abs()
            # OR for a full 0→1 change (can be huge on scaled data)
            coef["Odds Ratio (per +1.0)"] = np.exp(coef["Coefficient"]).round(4)
            # OR for a smaller +0.1 step (report-friendly)
            coef["Odds Ratio (per +0.1)"] = np.exp(coef["Coefficient"] * 0.1).round(4)
            coef["Direction"] = np.where(coef["Coefficient"] >= 0,
                                         "↑ increases Class 1 odds",
                                         "↓ decreases Class 1 odds")
            coef = coef.sort_values("|Coefficient|", ascending=False).reset_index(drop=True)
            st.markdown("**Logistic Regression — Coefficients & Odds Ratios**")
            st.dataframe(coef[["Feature","Coefficient","Odds Ratio (per +0.1)","Odds Ratio (per +1.0)","Direction"]])
            coef.to_csv("feature_importance_logistic.csv", index=False)
            st.caption("Saved: feature_importance_logistic.csv (includes both OR columns)")

        # Trees/Forest: standard Gini importance (% of total split gain)
        def show_tree_importance(model, features, label):
            imp = pd.DataFrame({
                "Feature": features,
                "Gini Importance": getattr(model, "feature_importances_", np.zeros(len(features)))
            }).sort_values("Gini Importance", ascending=False).reset_index(drop=True)
            imp["Gini Importance (%)]"] = (imp["Gini Importance"] * 100).round(2)
            st.markdown(f"**{label} — Gini Importance (%)**")
            st.dataframe(imp[["Feature","Gini Importance (%)]"]])
            fname = f"feature_importance_{label.lower().replace(' ','_')}.csv"
            imp.to_csv(fname, index=False)
            st.caption(f"Saved: {fname}")

        # Render for whichever models are loaded
        if "Logistic Regression" in models:
            try:
                show_logit_importance(models["Logistic Regression"], feature_cols)
            except Exception as e:
                st.warning(f"Could not compute logistic regression importances: {e}")

        if "Decision Tree" in models:
            show_tree_importance(models["Decision Tree"], feature_cols, "Decision Tree")

        if "Random Forest" in models:
            show_tree_importance(models["Random Forest"], feature_cols, "Random Forest")

        st.caption("Notes: Features are scaled 0–1. Odds Ratio (per +0.1) shows the effect for a small, realistic increase; "
                   "tree importances are relative contributions to splits (sum to 100%).")

