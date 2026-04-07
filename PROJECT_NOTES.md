# Project Notes for Internship Viva

## Project Title

GlucoTrack Analytics: Diabetes Risk Analysis and Early Prediction Dashboard

## Abstract

GlucoTrack Analytics is a healthcare analytics project created for GTU Problem Domain 5. The project studies diabetes-related patient data from a public Kaggle dataset and presents key patterns through a browser dashboard. It combines exploratory data analysis, simple machine learning, and live probability estimation to show how healthcare data can support early disease screening and decision-making.

## Problem Definition

Hospitals and clinics collect large volumes of patient information, but identifying meaningful risk patterns quickly is still difficult. Diabetes is strongly influenced by age, obesity, blood glucose, HbA1c level, hypertension, heart disease, and lifestyle factors. This project organizes those indicators into a single dashboard so that users can inspect the dataset, compare risk-related trends, and estimate the probability of diabetes for a new patient profile.

## Objectives

- To perform EDA on a public healthcare dataset
- To identify major indicators associated with diabetes
- To build a simple classification workflow for disease prediction
- To display model evaluation metrics in an understandable way
- To create a project that can be demonstrated easily during internship review or viva

## Dataset Used

- Source: Kaggle
- Dataset name: Diabetes Prediction Dataset
- Link: https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset

## Main Modules

### 1. Dataset Loading

The dashboard loads the Kaggle CSV from the local `data` folder or accepts a manual upload.

### 2. Exploratory Data Analysis

The app shows prevalence, average BMI, glucose trends, HbA1c-based separation, age distribution, smoking history distribution, and gender split.

### 3. Machine Learning Module

A logistic regression model is trained in the browser using the loaded dataset. The project uses a train/test split and calculates accuracy, precision, recall, F1-score, and confusion matrix.

### 4. Live Risk Prediction

Users can enter a new patient profile and the system estimates diabetes probability from the learned model weights.

## Why This Project Is Useful

- Demonstrates EDA on healthcare records
- Connects health indicators to a real prediction problem
- Shows practical use of machine learning in medical analytics
- Works without backend setup, so it is easy to deploy on GitHub
- Fits well for internship demonstration because both analytics and prediction are visible in one interface

## Limitations

- The model is a baseline browser implementation, not a hospital-grade clinical system
- The prediction output is for academic demonstration only
- The quality of results depends on the dataset loaded by the user
- No patient authentication or hospital database integration is included

## Future Scope

- Add Python-based training notebooks and model comparison
- Connect the system to hospital records or APIs
- Add downloadable reports for doctors and administrators
- Include more disease domains such as heart disease or stroke risk
- Extend to image-based or signal-based healthcare analysis in later versions

## Short Viva Explanation

This project fits GTU Domain 5 because it performs exploratory data analysis on a public patient dataset, identifies the health indicators that influence diabetes, builds a machine learning workflow for disease prediction, and presents the results in a dashboard that supports early screening and healthcare decision-making.
