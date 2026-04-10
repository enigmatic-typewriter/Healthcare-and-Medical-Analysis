from __future__ import annotations

from io import StringIO

import pandas as pd
import streamlit as st

from src.healthcare_analytics import (
    DATASET_PATH,
    MODEL_CATALOG,
    clean_dataframe,
    format_percent,
    risk_bucket,
    run_training_workflow,
)


st.set_page_config(page_title="GlucoTrack Analytics", page_icon="H", layout="wide")


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


def load_dataset(uploaded_file) -> tuple[pd.DataFrame, str]:
    if uploaded_file is not None:
        return clean_dataframe(pd.read_csv(uploaded_file)), f"Uploaded file: {uploaded_file.name}"
    if DATASET_PATH.exists():
        return clean_dataframe(pd.read_csv(DATASET_PATH)), f"Local file: {DATASET_PATH}"
    return clean_dataframe(build_demo_dataframe()), "Built-in demo dataset"


@st.cache_data(show_spinner=False)
def prepare_dataset(uploaded_file_bytes: bytes | None, uploaded_name: str | None):
    uploaded_file = None
    if uploaded_file_bytes is not None and uploaded_name is not None:
        uploaded_file = StringIO(uploaded_file_bytes.decode("utf-8"))
        uploaded_file.name = uploaded_name
    return load_dataset(uploaded_file)


@st.cache_resource(show_spinner=False)
def train_model(data: pd.DataFrame):
    return run_training_workflow(data)


def render_overview(data: pd.DataFrame, results: dict):
    scored_data = results["scored_data"]
    row_one = st.columns(4)
    row_one[0].metric("Patients", f"{len(data):,}")
    row_one[1].metric("Diabetes prevalence", format_percent((data["diabetes"] == 1).mean()))
    row_one[2].metric("High-risk share", format_percent((scored_data["risk_band"] == "High").mean()))
    row_one[3].metric("Immediate review", f"{results['operational_metrics']['immediate_review_patients']:,}")

    row_two = st.columns(4)
    row_two[0].metric("Average BMI", f"{data['bmi'].mean():.1f}")
    row_two[1].metric("Average HbA1c", f"{data['HbA1c_level'].mean():.1f}")
    row_two[2].metric("Average glucose", f"{data['blood_glucose_level'].mean():.0f}")
    row_two[3].metric("Comorbidity share", format_percent(((data["hypertension"] == 1) | (data["heart_disease"] == 1)).mean()))

    left, right = st.columns([1.15, 1])
    with left:
        st.subheader("Problem Context")
        st.write(
            "This project studies diabetes risk from patient health records and shows how analytics can support earlier screening. "
            "It combines exploratory analysis, multiple ML models, risk segmentation, and a simple operational planning view."
        )
        st.dataframe(results["cohort_summary"], use_container_width=True)

    with right:
        st.subheader("Key Takeaways")
        for takeaway in results["feature_takeaways"]:
            st.info(takeaway)


def render_eda(results: dict):
    scored_data = results["scored_data"]
    left, right = st.columns(2)

    with left:
        st.subheader("Population Trends")
        st.caption("Predicted risk distribution")
        st.bar_chart(results["risk_distribution"].set_index("risk_band")["patients"])
        st.caption("Age band vs predicted risk")
        age_risk = results["segment_summary"].set_index("age_band")
        st.bar_chart(age_risk)
        st.caption("Smoking history")
        st.bar_chart(scored_data["smoking_history"].value_counts())

    with right:
        st.subheader("Clinical Signals")
        st.caption("Glucose bands")
        st.bar_chart(scored_data["glucose_band"].value_counts())
        st.caption("BMI bands")
        st.bar_chart(scored_data["bmi_band"].value_counts())
        st.caption("Average indicators by diabetes class")
        st.dataframe(results["cohort_summary"], use_container_width=True)

    st.subheader("Priority Queue Preview")
    preview_cols = [
        "gender",
        "age",
        "bmi",
        "HbA1c_level",
        "blood_glucose_level",
        "predicted_probability",
        "risk_band",
        "triage_priority",
    ]
    st.dataframe(
        scored_data[preview_cols].head(15).style.format({"predicted_probability": "{:.2%}"}),
        use_container_width=True,
    )


def render_model_results(results: dict):
    st.subheader("Model Lab")
    metrics = results["best_metrics"]

    metric_cols = st.columns(5)
    metric_cols[0].metric("Selected model", results["best_model_name"])
    metric_cols[1].metric("Accuracy", format_percent(metrics["accuracy"]))
    metric_cols[2].metric("Precision", format_percent(metrics["precision"]))
    metric_cols[3].metric("Recall", format_percent(metrics["recall"]))
    metric_cols[4].metric("F1-score", format_percent(metrics["f1_score"]))

    left, middle, right = st.columns([1.2, 1, 1])
    with left:
        st.caption("Compared models")
        st.dataframe(
            results["leaderboard"].style.format(
                {
                    "accuracy": "{:.3f}",
                    "precision": "{:.3f}",
                    "recall": "{:.3f}",
                    "f1_score": "{:.3f}",
                    "roc_auc": "{:.3f}",
                }
            ),
            use_container_width=True,
        )

    with middle:
        matrix_frame = pd.DataFrame(
            metrics["confusion_matrix"],
            index=["Actual 0", "Actual 1"],
            columns=["Predicted 0", "Predicted 1"],
        )
        st.caption("Confusion matrix")
        st.dataframe(matrix_frame, use_container_width=True)

    with right:
        st.caption("Feature importance")
        st.bar_chart(results["feature_importance"].set_index("feature")["importance"])
        st.caption(
            f"{results['best_model_name']} scored highest on the held-out split, so it is used for the live prediction section."
        )


def render_screening_planner(results: dict):
    scored_data = results["scored_data"]
    metrics = results["operational_metrics"]

    st.subheader("Screening Planner")
    st.write(
        "This module is a simple academic add-on. It estimates how much follow-up load the hospital might face "
        "if the current dataset is treated as a screening batch."
    )

    capacity_col, rate_col = st.columns(2)
    daily_capacity = capacity_col.slider("Daily screening capacity", min_value=10, max_value=200, value=35, step=5)
    follow_up_rate = rate_col.slider("Moderate-risk follow-up rate", min_value=0.1, max_value=1.0, value=0.6, step=0.1)

    estimated_queue = metrics["high_risk_patients"] + int(metrics["moderate_risk_patients"] * follow_up_rate)
    days_needed = max(1, int((estimated_queue + daily_capacity - 1) / daily_capacity))
    admission_watch = int(
        len(
            scored_data[
                (scored_data["predicted_probability"] >= 0.75)
                & (scored_data["blood_glucose_level"] >= 180)
                & ((scored_data["hypertension"] == 1) | (scored_data["heart_disease"] == 1))
            ]
        )
    )

    plan_cols = st.columns(4)
    plan_cols[0].metric("High-risk cases", f"{metrics['high_risk_patients']:,}")
    plan_cols[1].metric("Expected follow-ups", f"{estimated_queue:,}")
    plan_cols[2].metric("Days to clear queue", f"{days_needed}")
    plan_cols[3].metric("Admission watchlist", f"{admission_watch:,}")

    st.caption("Triage distribution")
    triage_chart = results["triage_distribution"].set_index("triage_priority")["patients"]
    st.bar_chart(triage_chart)

    st.caption("Suggested operational interpretation")
    st.write(
        f"With a daily capacity of {daily_capacity} screenings, the current batch would need about {days_needed} day(s) "
        f"to clear. Around {admission_watch} patient(s) show both high predicted risk and added comorbidity signals."
    )


def render_predictor(results: dict):
    st.subheader("Live Patient Prediction")
    model = results["best_model"]

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
        st.error(f"Estimated diabetes probability: {format_percent(probability)}")
    elif probability >= 0.4:
        st.warning(f"Estimated diabetes probability: {format_percent(probability)}")
    else:
        st.success(f"Estimated diabetes probability: {format_percent(probability)}")

    detail_cols = st.columns(4)
    detail_cols[0].metric("Risk band", risk_bucket(probability))
    detail_cols[1].metric("Model used", results["best_model_name"])
    detail_cols[2].metric("Models compared", len(MODEL_CATALOG))
    detail_cols[3].metric("Glucose reading", f"{glucose}")

    st.caption("This output is only for academic demonstration and not for real medical diagnosis.")


def render_project_notes(results: dict, source: str):
    st.subheader("Project Notes")
    st.write(
        "The idea behind GlucoTrack Analytics is to create a mini healthcare analytics system that is simple to explain in viva "
        "but still shows data cleaning, EDA, model training, comparison, and decision-support style outputs."
    )

    left, right = st.columns(2)
    with left:
        st.markdown(
            """
            **Objectives**

            - perform exploratory data analysis on patient records
            - identify major health indicators linked with diabetes risk
            - compare multiple machine learning models
            - support early screening through a live prediction interface
            - simulate a basic operational planning view for hospital follow-up
            """
        )
    with right:
        st.markdown(
            f"""
            **Current Run**

            - dataset source: `{source}`
            - rows loaded: `{results['operational_metrics']['total_patients']}`
            - selected model: `{results['best_model_name']}`
            - immediate review cases: `{results['operational_metrics']['immediate_review_patients']}`
            - alert share: `{format_percent(results['operational_metrics']['alert_share'])}`
            """
        )

    st.markdown(
        """
        **Limitations**

        - this project uses a public dataset and should be treated as an academic prototype
        - no integration with hospital databases or medical devices is included
        - prediction quality depends completely on the uploaded dataset
        - model output should never be interpreted as clinical advice
        """
    )


def main():
    st.title("GlucoTrack Analytics")
    st.caption("Healthcare and Medical Analytics dashboard for diabetes risk analysis, triage support, and early screening")

    st.sidebar.header("Dataset Controls")
    uploaded = st.sidebar.file_uploader("Upload Kaggle CSV", type=["csv"])
    st.sidebar.markdown("[Dataset link](https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset)")
    st.sidebar.write(f"Expected path: `{DATASET_PATH}`")
    st.sidebar.write("If no file is available, the app falls back to a small inbuilt sample.")

    uploaded_bytes = uploaded.getvalue() if uploaded is not None else None
    uploaded_name = uploaded.name if uploaded is not None else None
    data, source = prepare_dataset(uploaded_bytes, uploaded_name)
    results = train_model(data)

    st.sidebar.success(source)
    st.sidebar.write(f"Rows loaded: {len(data):,}")
    st.sidebar.write(f"Selected model: {results['best_model_name']}")
    st.sidebar.write(f"High-risk patients: {results['operational_metrics']['high_risk_patients']}")

    tabs = st.tabs(["Overview", "EDA", "Model Lab", "Screening Planner", "Prediction", "Notes"])

    with tabs[0]:
        render_overview(data, results)
    with tabs[1]:
        render_eda(results)
    with tabs[2]:
        render_model_results(results)
    with tabs[3]:
        render_screening_planner(results)
    with tabs[4]:
        render_predictor(results)
    with tabs[5]:
        render_project_notes(results, source)

    st.divider()
    st.subheader("Dataset Preview")
    st.dataframe(results["scored_data"].head(20), use_container_width=True)


if __name__ == "__main__":
    main()
