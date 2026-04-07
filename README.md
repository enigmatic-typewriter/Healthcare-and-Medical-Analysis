# GlucoTrack Analytics

GlucoTrack Analytics is a Streamlit-based healthcare analytics project built for GTU Problem Domain 5: Healthcare and Medical Analytics. It focuses on diabetes risk analysis using the Kaggle dataset `iammustafatz/diabetes-prediction-dataset` and demonstrates exploratory data analysis, machine learning, and live prediction in a deployable Python app.

## What the project does

- loads the diabetes prediction CSV from Kaggle or a local upload
- cleans and normalizes patient attributes before modeling
- shows EDA panels for prevalence, age, BMI, HbA1c, glucose, smoking history, and gender mix
- trains a logistic regression model with a train/test split
- displays accuracy, precision, recall, and confusion matrix values
- predicts diabetes probability for a new patient profile
- runs as a Streamlit app, so it is easy to deploy on Streamlit Community Cloud

## Dataset

Kaggle dataset: [Diabetes Prediction Dataset](https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset)

Expected file name:

```text
data/diabetes_prediction_dataset.csv
```

Expected columns:

```text
gender, age, hypertension, heart_disease, smoking_history, bmi, HbA1c_level, blood_glucose_level, diabetes
```

If the Kaggle CSV is not present, the app falls back to a built-in demo dataset so the project still opens correctly.

## How to run

1. Clone the repository.
2. Create and activate a Python virtual environment.
3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Download the Kaggle dataset CSV.
5. Put the file inside the `data` folder with the name `diabetes_prediction_dataset.csv`.
6. Run the app:

```bash
streamlit run app.py
```

You can also skip step 5 and upload the CSV manually from the sidebar inside the app.

## Streamlit deployment

1. Push the repository to GitHub.
2. Open Streamlit Community Cloud.
3. Create a new app and select this repository.
4. Set the main file path to `app.py`.
5. Deploy.

## GTU objective mapping

- Performs EDA on patient health records
- Identifies key indicators related to diabetes risk
- Builds a disease prediction workflow using machine learning logic
- Supports early diagnosis and health decision-making through risk scoring
- Provides a clean internship-ready visualization layer for presentation and GitHub review

## Project structure

```text
Healthcare-and-Medical-Analytics/
|-- app.py
|-- requirements.txt
|-- README.md
|-- PROJECT_NOTES.md
|-- data/
```

## Future improvements

- Add downloadable reports and charts
- Compare multiple ML models instead of one logistic regression baseline
- Connect the interface to Flask, FastAPI, or a database backend
- Add role-specific dashboards for doctors and administrators
