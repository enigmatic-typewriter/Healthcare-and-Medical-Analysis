# Project Notes

## Project Title

GlucoTrack Analytics: Diabetes Risk Analysis and Screening Support System

## Abstract

GlucoTrack Analytics is a mini healthcare analytics system developed using patient health records related to diabetes. The project performs data cleaning, exploratory data analysis, machine learning model comparison, and live patient risk prediction. An additional screening planner module is included to estimate follow-up load and identify patients who may need immediate attention.

## Why I Chose This Problem

Diabetes is a common chronic disease and its risk is influenced by measurable factors such as age, BMI, blood glucose, HbA1c, hypertension, and heart disease. Because the problem is both medically important and data-friendly, it is a good fit for applying analytics and machine learning in a practical student project.

## Problem Statement

Hospitals and clinics generate large amounts of patient data, but useful insights are often delayed because the data is spread across records and requires manual review. This creates three practical challenges:

- early identification of high-risk patients
- understanding major clinical risk factors quickly
- planning follow-up effort when many patients need screening

## Objectives

- perform EDA on a healthcare dataset
- find the indicators that most strongly influence diabetes risk
- compare multiple machine learning models
- build a dashboard for early screening support
- add a simple planning view for patient follow-up management

## Dataset Used

- Platform: Kaggle
- Dataset: Diabetes Prediction Dataset
- Link: https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset

## Input Features Used

- gender
- age
- hypertension
- heart disease
- smoking history
- BMI
- HbA1c level
- blood glucose level

## Output

- diabetes class
- predicted diabetes probability
- risk band
- triage priority

## Project Modules

### 1. Data Loading and Cleaning

The application checks for required columns, standardizes labels, converts numeric fields, and removes unusable values.

### 2. Exploratory Data Analysis

This module shows:

- disease prevalence
- age and BMI segmentation
- smoking pattern analysis
- glucose and HbA1c trends
- diabetic vs non-diabetic patient comparison

### 3. Machine Learning Model Comparison

Three models are compared:

- Logistic Regression
- Random Forest
- Extra Trees

The best model is selected based on evaluation scores.

### 4. Patient Risk Prediction

The user enters patient details manually and the dashboard estimates the probability of diabetes.

### 5. Screening Planner

This module estimates:

- how many patients fall in high-risk or moderate-risk groups
- how many may need immediate review
- approximate screening days required based on daily capacity

## Important Observations

- HbA1c and blood glucose usually become the strongest predictors
- diabetic patients tend to have higher BMI and higher age
- comorbidities like hypertension and heart disease increase patient risk
- risk segmentation is useful for prioritizing follow-up instead of treating every patient equally

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit

## Limitations

- dataset quality directly affects prediction quality
- this is only an academic prototype
- no real hospital integration, authentication, or live EHR connection is included
- output should not be used as medical advice

## Future Scope

- add SHAP or other explainable AI techniques
- include readmission or heart disease prediction modules
- connect with a database or API for hospital records
- generate downloadable PDF or CSV reports for hospital review teams

## Short Viva Answer

This project belongs to Healthcare and Medical Analytics because it uses patient records to perform EDA, identifies risk factors for diabetes, compares machine learning models, and provides a dashboard for early screening and basic decision support.
