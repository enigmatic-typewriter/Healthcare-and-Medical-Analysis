# Project Notes for Internship Viva

## Project Title

GlucoTrack Analytics: Diabetes Risk Analysis and Early Prediction System

## Abstract

GlucoTrack Analytics is a healthcare analytics project that studies diabetes-related patient data using a publicly available Kaggle dataset. The project combines exploratory data analysis, machine learning model comparison, and a live prediction dashboard to show how health records can be used for early screening and decision support.

## Problem Statement

Healthcare institutions collect large amounts of patient information, but turning that data into quick and useful decisions is still difficult. Delayed diagnosis, increasing disease burden, and fragmented health data create a strong need for systems that can detect disease risk early and summarize important clinical trends clearly.

This project focuses on diabetes because it is strongly associated with measurable patient attributes such as age, BMI, HbA1c level, blood glucose level, hypertension, heart disease, and lifestyle habits.

## Objectives

- perform EDA on a healthcare dataset
- identify the indicators that are most associated with diabetes risk
- train and compare multiple machine learning models
- build a prediction interface for individual patient screening
- create a project that can be demonstrated easily during internship evaluation or viva

## Dataset Used

- Source: Kaggle
- Name: Diabetes Prediction Dataset
- Link: https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset

## Main Modules

### 1. Data Ingestion and Cleaning

The system reads the Kaggle CSV, validates the expected columns, standardizes categorical values, and handles missing values before analysis.

### 2. Exploratory Data Analysis

The dashboard displays class distribution, smoking history, age bands, average clinical indicators, gender-wise outcomes, and high-risk patient segmentation.

### 3. Machine Learning Pipeline

The project compares Logistic Regression, Random Forest, and Extra Trees classifiers. The best model is selected based on validation performance and used for live prediction.

### 4. Patient Risk Prediction

Users can enter patient details manually and the app returns an estimated diabetes probability along with a low, moderate, or high risk category.

### 5. Offline Training Report

A separate training script exports the model leaderboard, feature importance table, and a training summary file so the project is reproducible outside the dashboard.

## Key Indicators Observed

- HbA1c level
- blood glucose level
- BMI
- age
- hypertension
- heart disease
- smoking history

## Tools and Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit

## Usefulness of the Project

- helps demonstrate healthcare EDA in a practical format
- shows how machine learning can support early disease risk screening
- presents both technical implementation and business relevance
- is simple to run locally and easy to explain during a presentation

## Limitations

- results depend on the quality and coverage of the dataset
- the system is an academic prototype, not a medical product
- no hospital integration, authentication, or real-time patient record connection is included

## Future Scope

- add explainable AI visualizations for individual predictions
- expand into heart disease or hospital readmission prediction
- connect the dashboard to APIs or database-backed health systems
- include downloadable PDF summaries and model monitoring

## One-Line Viva Summary

This project fits Healthcare and Medical Analytics because it performs EDA on patient records, identifies risk factors, compares ML models for disease prediction, and provides a dashboard that supports early diagnosis and healthcare decision-making.
