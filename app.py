from __future__ import annotations

from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


st.set_page_config(
    page_title="GlucoTrack Analytics",
    page_icon="H",
    layout="wide",
)


DATASET_PATH = Path("data/diabetes_prediction_dataset.csv")
REQUIRED_COLUMNS = [
    "gender",
    "age",
    "hypertension",
    "heart_disease",
    "smoking_history",
    "bmi",
    "HbA1c_level",
    "blood_glucose_level",
    "diabetes",
]

NUMERIC_COLUMNS = [
    "age",
    "hypertension",
    "heart_disease",
    "bmi",
    "HbA1c_level",
    "blood_glucose_level",
]

CATEGORICAL_COLUMNS = ["gender", "smoking_history"]


def build_demo_dataframe() -> pd.DataFrame:
    raw = """gender,age,hypertension,heart_disease,smoking_history,bmi,HbA1c_level,blood_glucose_level,diabetes
Female,42,0,0,never,24.1,5.4,112,0
Male,58,1,0,former,31.8,7.2,190,1
Female,36,0,0,No Info,22.7,5.1,98,0
Male,61,1,1,current,34.5,8.1,224,1
Female,49,0,0,former,29.2,6.4,166,1
Male,29,0,0,never,23.8,4.9,96,0
Female,67,1,1,not current,33.4,7.8,210,1
Male,45,0,0,ever,28.1,5.8,134,0
Female,53,1,0,former,32.2,6.9,184,1
Male,39,0,0,never,25.6,5.3,110,0
Female,31,0,0,never,21.9,4.8,92,0
Male,72,1,1,former,30.7,8.4,236,1
Female,64,1,0,current,35.1,7.6,206,1
Male,51,0,0,not current,27.2,5.9,142,0
Female,47,0,0,ever,26.8,5.7,128,0
Male,55,1,0,former,33.5,7.1,188,1
Female,26,0,0,never,20.3,4.7,88,0
Male,43,0,0,current,29.4,6.1,156,1
Female,59,1,0,former,31.6,7.0,179,1
Male,34,0,0,never,24.7,5.0,101,0
Female,62,1,1,not current,36.4,8.2,230,1
Male,41,0,0,ever,27.5,5.6,126,0
Female,57,1,0,former,30.8,6.8,174,1
Male,38,0,0,No Info,26.1,5.2,108,0
Female,44,0,0,never,23.4,5.1,105,0
Male,66,1,1,current,35.8,8.0,218,1
Female,52,1,0,former,29.7,6.6,170,1
Male,28,0,0,never,22.9,4.8,94,0
Female,48,0,0,ever,27.9,5.9,138,0
Male,63,1,1,former,34.2,7.7,208,1"""
    return pd.read_csv(StringIO(raw))


def clean_dataframe(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    data.columns = [column.strip() for column in data.columns]

    missing = [column for column in REQUIRED_COLUMNS if column not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    data = data[REQUIRED_COLUMNS].copy()

    data["gender"] = data["gender"].fillna("Other").astype(str).str.strip().replace({"male": "Male", "female": "Female"})
    data["gender"] = data["gender"].replace({"Male": "Male", "Female": "Female"}).where(
        data["gender"].isin(["Male", "Female"]),
        "Other",
    )

    data["smoking_history"] = data["smoking_history"].fillna("No Info").astype(str).str.strip()
    data["smoking_history"] = data["smoking_history"].replace({"nan": "No Info", "N/A": "No Info"})

    for column in NUMERIC_COLUMNS + ["diabetes"]:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    data = data.dropna(subset=["diabetes"]).copy()
    data["diabetes"] = data["diabetes"].astype(int)

    return data


def load_dataset(uploaded_file) -> tuple[pd.DataFrame, str]:
    if uploaded_file is not None:
        frame = pd.read_csv(uploaded_file)
        return clean_dataframe(frame), f"Uploaded file: {uploaded_file.name}"

    if DATASET_PATH.exists():
        frame = pd.read_csv(DATASET_PATH)
        return clean_dataframe(frame), f"Local file: {DATASET_PATH}"

    return clean_dataframe(build_demo_dataframe()), "Built-in demo dataset"


@st.cache_data(show_spinner=False)
def prepare_dataset(uploaded_file_bytes: bytes | None, uploaded_name: str | None):
    uploaded_file = None
    if uploaded_file_bytes is not None and uploaded_name is not None:
        uploaded_file = StringIO(uploaded_file_bytes.decode("utf-8"))
        uploaded_file.name = uploaded_name

    data, source = load_dataset(uploaded_file)
    return data, source


@st.cache_resource(show_spinner=False)
def train_model(data: pd.DataFrame):
    features = data.drop(columns=["diabetes"])
    target = data["diabetes"]

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=42,
        stratify=target if target.nunique() > 1 else None,
    )

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_COLUMNS),
            ("cat", categorical_pipeline, CATEGORICAL_COLUMNS),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=1000)),
        ]
    )

    model.fit(x_train, y_train)
    predictions = model.predict(x_test)

    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1_score": f1_score(y_test, predictions, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, predictions, labels=[0, 1]),
    }

    feature_names = model.named_steps["preprocessor"].get_feature_names_out()
    weights = model.named_steps["classifier"].coef_[0]
    importance = (
        pd.DataFrame({"feature": feature_names, "weight": weights, "abs_weight": np.abs(weights)})
        .sort_values("abs_weight", ascending=False)
        .head(8)
    )

    return model, metrics, importance


def glucose_band(glucose_value: float) -> str:
    if glucose_value < 100:
        return "Normal"
    if glucose_value < 140:
        return "Elevated"
    if glucose_value < 200:
        return "High"
    return "Critical"


def format_pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def render_overview(data: pd.DataFrame):
    prevalence = (data["diabetes"] == 1).mean()
    hypertension_share = (data["hypertension"] == 1).mean()
    high_hba1c = (data["HbA1c_level"] >= 6.5).sum()
    critical_glucose = (data["blood_glucose_level"] >= 200).sum()

    row_one = st.columns(4)
    row_one[0].metric("Patients loaded", f"{len(data):,}")
    row_one[1].metric("Diabetes prevalence", format_pct(prevalence))
    row_one[2].metric("Average BMI", f"{data['bmi'].mean():.1f}")
    row_one[3].metric("Average glucose", f"{data['blood_glucose_level'].mean():.0f}")

    row_two = st.columns(4)
    row_two[0].metric("Hypertension share", format_pct(hypertension_share))
    row_two[1].metric("High HbA1c cases", f"{high_hba1c:,}")
    row_two[2].metric("Senior population", format_pct((data["age"] >= 60).mean()))
    row_two[3].metric("Critical glucose cases", f"{critical_glucose:,}")


def render_cohort_insights(data: pd.DataFrame):
    positive = data[data["diabetes"] == 1]
    negative = data[data["diabetes"] == 0]

    st.subheader("Cohort Snapshot")

    col1, col2, col3 = st.columns(3)
    col1.info(
        f"Patients with diabetes are older on average ({positive['age'].mean():.1f} years) "
        f"than non-diabetic patients ({negative['age'].mean():.1f} years)."
    )
    col2.info(
        f"The diabetic group has a higher average BMI ({positive['bmi'].mean():.1f}) "
        f"than the non-diabetic group ({negative['bmi'].mean():.1f})."
    )
    col3.info(
        f"HbA1c shows strong separation in this dataset: {negative['HbA1c_level'].mean():.1f} "
        f"for non-diabetic records versus {positive['HbA1c_level'].mean():.1f} for diabetic records."
    )


def render_eda(data: pd.DataFrame):
    st.subheader("Exploratory Data Analysis")

    eda_col1, eda_col2 = st.columns(2)

    with eda_col1:
        outcome_counts = data["diabetes"].map({0: "Non-diabetic", 1: "Diabetic"}).value_counts()
        st.caption("Class balance")
        st.bar_chart(outcome_counts)

        gender_counts = data["gender"].value_counts()
        st.caption("Gender distribution")
        st.bar_chart(gender_counts)

        glucose_counts = data["blood_glucose_level"].apply(glucose_band).value_counts()
        st.caption("Glucose bands")
        st.bar_chart(glucose_counts)

    with eda_col2:
        smoking_counts = data["smoking_history"].value_counts()
        st.caption("Smoking history")
        st.bar_chart(smoking_counts)

        numeric_summary = (
            data.groupby("diabetes")[["age", "bmi", "HbA1c_level", "blood_glucose_level"]]
            .mean()
            .rename(index={0: "Non-diabetic", 1: "Diabetic"})
        )
        st.caption("Average clinical indicators by class")
        st.dataframe(numeric_summary, use_container_width=True)


def render_model_results(metrics: dict, importance: pd.DataFrame):
    st.subheader("Model Evaluation")

    metric_cols = st.columns(4)
    metric_cols[0].metric("Accuracy", format_pct(metrics["accuracy"]))
    metric_cols[1].metric("Precision", format_pct(metrics["precision"]))
    metric_cols[2].metric("Recall", format_pct(metrics["recall"]))
    metric_cols[3].metric("F1-score", format_pct(metrics["f1_score"]))

    matrix = metrics["confusion_matrix"]
    left, right = st.columns([1, 1.3])

    with left:
        matrix_frame = pd.DataFrame(
            matrix,
            index=["Actual 0", "Actual 1"],
            columns=["Predicted 0", "Predicted 1"],
        )
        st.caption("Confusion matrix")
        st.dataframe(matrix_frame, use_container_width=True)

    with right:
        chart_frame = importance.set_index("feature")[["abs_weight"]]
        st.caption("Top feature weights")
        st.bar_chart(chart_frame)


def render_predictor(model: Pipeline):
    st.subheader("Live Prediction")

    col1, col2, col3 = st.columns(3)
    gender = col1.selectbox("Gender", ["Female", "Male", "Other"])
    age = col2.number_input("Age", min_value=1, max_value=100, value=48)
    smoking_history = col3.selectbox("Smoking history", ["never", "former", "current", "not current", "ever", "No Info"])

    col4, col5, col6 = st.columns(3)
    hypertension = col4.selectbox("Hypertension", [0, 1], format_func=lambda value: "Yes" if value == 1 else "No")
    heart_disease = col5.selectbox("Heart disease", [0, 1], format_func=lambda value: "Yes" if value == 1 else "No")
    bmi = col6.number_input("BMI", min_value=10.0, max_value=80.0, value=29.8, step=0.1)

    col7, col8 = st.columns(2)
    hba1c = col7.number_input("HbA1c level", min_value=3.0, max_value=15.0, value=6.8, step=0.1)
    glucose = col8.number_input("Blood glucose level", min_value=60, max_value=350, value=182, step=1)

    candidate = pd.DataFrame(
        [
            {
                "gender": gender,
                "age": age,
                "hypertension": hypertension,
                "heart_disease": heart_disease,
                "smoking_history": smoking_history,
                "bmi": bmi,
                "HbA1c_level": hba1c,
                "blood_glucose_level": glucose,
            }
        ]
    )

    probability = float(model.predict_proba(candidate)[0][1])

    if probability >= 0.7:
        st.error(f"Estimated diabetes probability: {format_pct(probability)}")
    elif probability >= 0.4:
        st.warning(f"Estimated diabetes probability: {format_pct(probability)}")
    else:
        st.success(f"Estimated diabetes probability: {format_pct(probability)}")

    st.caption("This prediction is for academic demonstration only and should not be used as a clinical diagnosis.")


def main():
    st.title("GlucoTrack Analytics")
    st.caption("Diabetes risk analysis and early prediction dashboard for GTU Problem Domain 5")

    st.sidebar.header("Dataset")
    uploaded = st.sidebar.file_uploader("Upload Kaggle CSV", type=["csv"])
    st.sidebar.markdown(
        "[Dataset link](https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset)"
    )

    uploaded_bytes = uploaded.getvalue() if uploaded is not None else None
    uploaded_name = uploaded.name if uploaded is not None else None
    data, source = prepare_dataset(uploaded_bytes, uploaded_name)

    st.sidebar.success(source)
    st.sidebar.write(f"Rows loaded: {len(data):,}")

    model, metrics, importance = train_model(data)

    render_overview(data)
    st.divider()
    render_cohort_insights(data)
    st.divider()
    render_eda(data)
    st.divider()
    render_model_results(metrics, importance)
    st.divider()
    render_predictor(model)
    st.divider()

    st.subheader("Data Preview")
    st.dataframe(data.head(20), use_container_width=True)


if __name__ == "__main__":
    main()
