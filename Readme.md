# 🏠 House Price Prediction using Machine Learning

A production-ready **Machine Learning web application** for predicting house prices based on housing features.  
The application is built using **Python, Scikit-learn, XGBoost**, and deployed with **Streamlit**.

This project demonstrates an **end-to-end ML workflow** including data preprocessing, model training, evaluation, and interactive prediction through a web interface.

---

## 📌 Project Overview

- Predicts house prices using structured numerical data
- Implements multiple regression models for comparison
- Provides real-time predictions through a Streamlit UI
- Displays model evaluation metrics and visualization

---

## 🚀 Key Features

- Interactive **Streamlit Web Application**
- Multiple Machine Learning Models:
  - Linear Regression
  - Random Forest Regressor
  - XGBoost Regressor
- User-selectable model comparison
- **R² Score** for model evaluation
- **Actual vs Predicted** visualization
- Clean and modular project structure

---

## 📊 Dataset Information

- **Dataset Name:** California Housing Dataset
- **Target Variable:** `median_house_value`
- **Features Include:**
  - Longitude, Latitude
  - Housing Median Age
  - Total Rooms & Bedrooms
  - Population & Households
  - Median Income

- File Location:
- data/cleaned_housing.csv
- 
---

## 🧠 Machine Learning Pipeline

1. Data Loading & Cleaning  
2. Missing Value Handling  
3. Feature Scaling  
4. Train-Test Split  
5. Model Training  
6. Model Evaluation (R² Score)  
7. Model Serialization using Joblib  
8. Streamlit Deployment  

---

## 🗂️ Project Structure

house_price/
│
├── app.py # Streamlit application
├── data/
│ └── cleaned_housing.csv # Preprocessed dataset
├── models/
│ ├── linear_model.pkl
│ ├── random_forest_model.pkl
│ └── xgboost_model.pkl
├── src/
│ └── train_model.py # Model training script
├── requirements.txt
├── environment.yml # Conda environment
└── README.md

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/jyoti-rawat1/house_price.git
cd house_price

Run Streamlit Application
streamlit run app.py
