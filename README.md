# GlucoTrack Analytics

GlucoTrack Analytics is a frontend healthcare analytics project built for GTU Problem Domain 5: Healthcare and Medical Analytics. It focuses on diabetes risk analysis using the Kaggle dataset `iammustafatz/diabetes-prediction-dataset` and demonstrates exploratory data analysis, lightweight machine learning, and a live prediction workflow inside a simple browser-based dashboard.

## What the project does

- loads the diabetes prediction CSV from Kaggle or a local upload
- cleans and normalizes patient attributes in the browser
- shows EDA panels for prevalence, age, BMI, HbA1c, glucose, smoking history, and gender mix
- trains a logistic regression model with a train/test split in the browser
- displays accuracy, precision, recall, and confusion matrix values
- predicts diabetes probability for a new patient profile
- runs as a static project, so it is easy to host on GitHub Pages

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
2. Download the Kaggle dataset CSV.
3. Put the file inside the `data` folder with the name `diabetes_prediction_dataset.csv`.
4. Open `index.html` in a browser.

You can also skip step 3 and upload the CSV manually from the dashboard.

## GTU objective mapping

- Performs EDA on patient health records
- Identifies key indicators related to diabetes risk
- Builds a disease prediction workflow using machine learning logic
- Supports early diagnosis and health decision-making through risk scoring
- Provides a clean internship-ready visualization layer for presentation and GitHub review

## Project structure

```text
Healthcare-and-Medical-Analytics/
|-- index.html
|-- styles.css
|-- app.js
|-- README.md
|-- PROJECT_NOTES.md
|-- data/
```

## Future improvements

- Move model training to Python for stronger experimentation
- Add downloadable reports and charts
- Compare multiple ML models instead of one logistic regression baseline
- Connect the interface to Flask, FastAPI, or a database backend
- Add role-specific dashboards for doctors and administrators
