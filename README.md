# GlucoTrack Analytics

GlucoTrack Analytics is an internship project under the domain **Healthcare & Medical Analytics**. The project focuses on early diabetes risk analysis using patient health records and combines:

- exploratory data analysis
- machine learning model comparison
- patient-level risk prediction
- a simple screening and follow-up planning module

The main idea is to show how healthcare data can be used not only for disease prediction, but also for decision support and screening prioritization.

## Problem Statement

Healthcare systems collect a large amount of patient data, but making practical use of that data is still difficult. Manual review can delay diagnosis, high-risk patients may not be prioritized quickly enough, and hospitals often need a simple way to estimate follow-up workload.

This project uses a diabetes prediction dataset to analyze major risk factors such as:

- age
- BMI
- HbA1c level
- blood glucose level
- hypertension
- heart disease
- smoking history

## Project Objectives

- perform EDA on patient health records
- identify key indicators linked with diabetes
- compare multiple ML models and choose the best one
- estimate diabetes probability for a new patient record
- simulate a basic screening workflow for hospital follow-up planning

## Dataset

- Source: [Kaggle - Diabetes Prediction Dataset](https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset)
- Expected path: `data/diabetes_prediction_dataset.csv`

Expected columns:

```text
gender, age, hypertension, heart_disease, smoking_history, bmi, HbA1c_level, blood_glucose_level, diabetes
```

If the Kaggle file is not available, the app automatically uses a built-in sample dataset so the project can still be demonstrated.

## Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit

## Main Modules

### 1. Data Cleaning

The project validates the dataset, standardizes labels, converts numeric fields, and removes unusable records.

### 2. Exploratory Data Analysis

The dashboard shows:

- prevalence and cohort metrics
- age-wise and risk-wise segmentation
- smoking history distribution
- BMI and glucose band analysis
- diabetic vs non-diabetic clinical summary

### 3. Model Comparison

The project compares:

- Logistic Regression
- Random Forest
- Extra Trees

The best-performing model is selected using validation metrics such as accuracy, precision, recall, F1-score, and ROC-AUC.

### 4. Live Prediction

A user can enter patient values manually and the app returns:

- estimated diabetes probability
- risk band
- model used for prediction

### 5. Screening Planner

This section is included to make the project more practical. Based on predicted risk groups, it estimates:

- high-risk cases
- expected follow-up load
- approximate days required to clear the queue
- patients needing immediate review

## Project Structure

```text
New project/
|-- app.py
|-- train_model.py
|-- src/
|   |-- healthcare_analytics.py
|-- data/
|   |-- .gitkeep
|-- artifacts/                # generated after training script
|-- index.html                # static web demo version
|-- app.js
|-- styles.css
|-- README.md
|-- PROJECT_NOTES.md
|-- requirements.txt
```

## How to Run

Install the dependencies:

```bash
pip install -r requirements.txt
```

Run the dashboard:

```bash
streamlit run app.py
```

Run the offline training report:

```bash
python train_model.py
```

## Generated Artifacts

The training script exports:

- `artifacts/model_leaderboard.csv`
- `artifacts/feature_importance.csv`
- `artifacts/risk_distribution.csv`
- `artifacts/triage_distribution.csv`
- `artifacts/cohort_summary.csv`
- `artifacts/training_summary.txt`

## Why This Project Is Good For Internship Review

- It clearly maps to the healthcare analytics problem statement.
- It includes both EDA and machine learning instead of only a prediction form.
- It is simple to run locally and easy to explain during viva or demo.
- It looks like a practical student project with both technical and application-level thinking.

## Disclaimer

This is an academic project created for internship demonstration. It is not a medical product and should not be used for actual diagnosis or treatment decisions.
