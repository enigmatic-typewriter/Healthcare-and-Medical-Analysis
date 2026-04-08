# GlucoTrack Analytics

GlucoTrack Analytics is a healthcare analytics project built for Problem Domain 5: Healthcare and Medical Analytics. It analyzes diabetes-related patient records from the Kaggle dataset `iammustafatz/diabetes-prediction-dataset`, performs exploratory data analysis, compares multiple machine learning models, and provides a Streamlit dashboard for risk prediction.

## Project Highlights

- exploratory analysis of patient demographics and clinical indicators
- identification of major diabetes risk factors such as age, BMI, HbA1c, glucose, hypertension, and heart disease
- comparison of multiple classification models instead of relying on a single baseline
- live patient risk scoring through a clean Streamlit interface
- reproducible training script that exports model comparison and feature-importance files

## Dataset

- Source: [Kaggle - Diabetes Prediction Dataset](https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset)
- Expected file location: `data/diabetes_prediction_dataset.csv`

Expected columns:

```text
gender, age, hypertension, heart_disease, smoking_history, bmi, HbA1c_level, blood_glucose_level, diabetes
```

If the Kaggle CSV is not available locally, the project uses a small built-in demo dataset so the app can still be opened and demonstrated.

## Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit

## Project Structure

```text
Healthcare-and-Medical-Analytics/
|-- app.py
|-- train_model.py
|-- src/
|   |-- healthcare_analytics.py
|-- data/
|   |-- .gitkeep
|-- README.md
|-- PROJECT_NOTES.md
|-- requirements.txt
```

## How to Run

1. Clone the repository.
2. Create and activate a virtual environment.
3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Download the Kaggle CSV and place it in the `data` folder.
5. Start the Streamlit app:

```bash
streamlit run app.py
```

## Training Workflow

To generate a quick offline training summary and export comparison artifacts:

```bash
python train_model.py
```

This creates:

- `artifacts/model_leaderboard.csv`
- `artifacts/feature_importance.csv`
- `artifacts/training_summary.txt`

## Dashboard Modules

- Overview KPIs for prevalence, glucose, BMI, HbA1c, and high-risk patient share
- Cohort insights comparing diabetic and non-diabetic records
- EDA charts for class balance, smoking history, age bands, and risk segments
- Model evaluation panel with leaderboard, confusion matrix, and feature importance
- Patient risk predictor for manual profile entry

## Why This Project Works For Internship Review

- It covers both analytics and machine learning in one project.
- The code is organized into a reusable training module and a presentation-friendly app.
- The dashboard is easy to demo, and the training script shows reproducible ML work beyond the UI.
- The project directly maps to real healthcare problems like early risk detection and data-driven screening.

## Disclaimer

This project is built for academic and internship demonstration purposes only. It is not a clinical decision support system and should not be used for diagnosis or treatment planning.
