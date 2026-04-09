from __future__ import annotations

from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


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
NUMERIC_COLUMNS = ["age", "hypertension", "heart_disease", "bmi", "HbA1c_level", "blood_glucose_level"]
CATEGORICAL_COLUMNS = ["gender", "smoking_history"]
MODEL_CATALOG = {
    "Logistic Regression": LogisticRegression(max_iter=1200, class_weight="balanced"),
    "Random Forest": RandomForestClassifier(
        n_estimators=250,
        max_depth=12,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced_subsample",
    ),
    "Extra Trees": ExtraTreesClassifier(
        n_estimators=300,
        max_depth=14,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    ),
}


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
Male,63,1,1,former,34.2,7.7,208,1
Female,35,0,0,never,24.6,5.0,99,0
Male,54,1,0,not current,32.8,6.9,181,1
Female,69,1,1,current,37.2,8.5,242,1
Male,46,0,0,former,28.7,6.0,148,0
Female,40,0,0,No Info,25.1,5.4,116,0
Male,60,1,1,ever,33.1,7.4,194,1"""
    return pd.read_csv(StringIO(raw))


def clean_dataframe(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    data.columns = [column.strip() for column in data.columns]
    missing = [column for column in REQUIRED_COLUMNS if column not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    data = data[REQUIRED_COLUMNS].copy()
    data["gender"] = data["gender"].fillna("Other").astype(str).str.strip().replace({"male": "Male", "female": "Female"})
    data["gender"] = data["gender"].where(data["gender"].isin(["Male", "Female"]), "Other")
    data["smoking_history"] = (
        data["smoking_history"].fillna("No Info").astype(str).str.strip().replace({"nan": "No Info", "N/A": "No Info"})
    )

    for column in NUMERIC_COLUMNS + ["diabetes"]:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    data = data.dropna(subset=["diabetes"]).copy()
    data["diabetes"] = data["diabetes"].astype(int)
    return data


def load_default_dataset() -> tuple[pd.DataFrame, str]:
    if DATASET_PATH.exists():
        return clean_dataframe(pd.read_csv(DATASET_PATH)), f"Local file: {DATASET_PATH}"
    return clean_dataframe(build_demo_dataframe()), "Built-in demo dataset"


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]),
                NUMERIC_COLUMNS,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                CATEGORICAL_COLUMNS,
            ),
        ]
    )


def format_percent(value: float) -> str:
    return f"{value * 100:.1f}%"


def risk_bucket(probability: float) -> str:
    if probability >= 0.7:
        return "High"
    if probability >= 0.4:
        return "Moderate"
    return "Low"


def glucose_band(glucose_value: float) -> str:
    if glucose_value < 100:
        return "Normal"
    if glucose_value < 140:
        return "Elevated"
    if glucose_value < 200:
        return "High"
    return "Critical"


def bmi_band(bmi_value: float) -> str:
    if bmi_value < 18.5:
        return "Underweight"
    if bmi_value < 25:
        return "Healthy"
    if bmi_value < 30:
        return "Overweight"
    return "Obese"


def age_band(age_value: float) -> str:
    if age_value < 25:
        return "Below 25"
    if age_value < 40:
        return "25-39"
    if age_value < 55:
        return "40-54"
    if age_value < 70:
        return "55-69"
    return "70+"


def triage_priority(row: pd.Series) -> str:
    if row["predicted_probability"] >= 0.8 or (
        row["blood_glucose_level"] >= 200 and row["HbA1c_level"] >= 6.8
    ):
        return "Immediate Review"
    if row["predicted_probability"] >= 0.55 or row["hypertension"] == 1 or row["heart_disease"] == 1:
        return "Follow-up Needed"
    return "Routine Monitoring"


def build_takeaways(feature_importance: pd.DataFrame, data: pd.DataFrame) -> list[str]:
    top_features = feature_importance["feature"].head(3).tolist()
    takeaways: list[str] = []

    for feature in top_features:
        if feature == "HbA1c_level":
            takeaways.append(
                "HbA1c is the strongest signal in this dataset, which matches its direct clinical relevance for diabetes screening."
            )
        elif feature == "blood_glucose_level":
            takeaways.append(
                "Blood glucose separates positive and negative cases clearly, so it becomes one of the main drivers of the model."
            )
        elif feature == "bmi":
            takeaways.append("BMI contributes strongly, suggesting obesity-linked risk is visible in the patient cohort.")
        elif feature == "age":
            takeaways.append("Age remains a meaningful factor, with diabetic patients tending to be older in the loaded sample.")
        elif feature == "hypertension":
            takeaways.append("Hypertension adds useful signal, reinforcing the role of comorbidities in diabetes risk assessment.")
        elif feature == "heart_disease":
            takeaways.append("Heart disease appears in many higher-risk records and improves patient risk stratification.")

    if not takeaways:
        takeaways.append(
            f"The average age of diabetic patients is {data.loc[data['diabetes'] == 1, 'age'].mean():.1f} years, higher than the non-diabetic group."
        )

    return takeaways


def run_training_workflow(data: pd.DataFrame) -> dict:
    x = data.drop(columns=["diabetes"])
    y = data["diabetes"]
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y if y.nunique() > 1 else None,
    )

    trained_models: dict[str, Pipeline] = {}
    model_metrics: dict[str, dict] = {}
    leaderboard_rows: list[dict] = []

    for model_name, estimator in MODEL_CATALOG.items():
        pipeline = Pipeline([("preprocessor", build_preprocessor()), ("classifier", estimator)])
        pipeline.fit(x_train, y_train)
        predicted = pipeline.predict(x_test)
        probabilities = pipeline.predict_proba(x_test)[:, 1]

        metrics = {
            "accuracy": accuracy_score(y_test, predicted),
            "precision": precision_score(y_test, predicted, zero_division=0),
            "recall": recall_score(y_test, predicted, zero_division=0),
            "f1_score": f1_score(y_test, predicted, zero_division=0),
            "roc_auc": roc_auc_score(y_test, probabilities) if y_test.nunique() > 1 else np.nan,
            "confusion_matrix": confusion_matrix(y_test, predicted, labels=[0, 1]),
        }
        trained_models[model_name] = pipeline
        model_metrics[model_name] = metrics
        leaderboard_rows.append({"model": model_name, **{key: value for key, value in metrics.items() if key != "confusion_matrix"}})

    leaderboard = pd.DataFrame(leaderboard_rows).sort_values(by=["f1_score", "roc_auc", "precision"], ascending=False)
    best_model_name = leaderboard.iloc[0]["model"]
    best_model = trained_models[best_model_name]

    scored_data = data.copy()
    scored_data["predicted_probability"] = best_model.predict_proba(x)[:, 1]
    scored_data["risk_band"] = scored_data["predicted_probability"].apply(risk_bucket)
    scored_data["glucose_band"] = scored_data["blood_glucose_level"].apply(glucose_band)
    scored_data["bmi_band"] = scored_data["bmi"].apply(bmi_band)
    scored_data["age_band"] = scored_data["age"].apply(age_band)
    scored_data["triage_priority"] = scored_data.apply(triage_priority, axis=1)

    importance = permutation_importance(
        best_model,
        x_test,
        y_test,
        n_repeats=4,
        random_state=42,
        scoring="f1",
        n_jobs=1,
    )
    feature_importance = (
        pd.DataFrame({"feature": x_test.columns, "importance": importance.importances_mean})
        .sort_values("importance", ascending=False)
        .head(8)
        .reset_index(drop=True)
    )

    cohort_summary = (
        data.groupby("diabetes")[["age", "bmi", "HbA1c_level", "blood_glucose_level", "hypertension", "heart_disease"]]
        .mean()
        .rename(index={0: "Non-diabetic", 1: "Diabetic"})
        .round(2)
    )
    risk_distribution = (
        scored_data["risk_band"]
        .value_counts()
        .rename_axis("risk_band")
        .reset_index(name="patients")
        .assign(share=lambda frame: frame["patients"] / len(scored_data))
    )
    triage_distribution = (
        scored_data["triage_priority"]
        .value_counts()
        .rename_axis("triage_priority")
        .reset_index(name="patients")
        .assign(share=lambda frame: frame["patients"] / len(scored_data))
    )
    segment_summary = (
        scored_data.pivot_table(
            index="age_band",
            columns="risk_band",
            values="predicted_probability",
            aggfunc="count",
            fill_value=0,
        )
        .reset_index()
    )

    operational_metrics = {
        "total_patients": len(scored_data),
        "high_risk_patients": int((scored_data["risk_band"] == "High").sum()),
        "moderate_risk_patients": int((scored_data["risk_band"] == "Moderate").sum()),
        "immediate_review_patients": int((scored_data["triage_priority"] == "Immediate Review").sum()),
        "follow_up_patients": int((scored_data["triage_priority"] == "Follow-up Needed").sum()),
        "mean_probability": float(scored_data["predicted_probability"].mean()),
        "alert_share": float((scored_data["blood_glucose_level"] >= 200).mean()),
    }

    return {
        "best_model_name": best_model_name,
        "best_model": best_model,
        "best_metrics": model_metrics[best_model_name],
        "leaderboard": leaderboard.reset_index(drop=True),
        "feature_importance": feature_importance,
        "scored_data": scored_data.sort_values("predicted_probability", ascending=False).reset_index(drop=True),
        "cohort_summary": cohort_summary,
        "risk_distribution": risk_distribution,
        "triage_distribution": triage_distribution,
        "segment_summary": segment_summary,
        "operational_metrics": operational_metrics,
        "feature_takeaways": build_takeaways(feature_importance, data),
    }
