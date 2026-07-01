
# Property Price Prediction using Linear Regression

## Description
Machine learning project to predict property prices using regression techniques with extensive data preprocessing and feature engineering.

👩‍💻 Author
Nikita Porwal
Data Scientist | Analytics & AI
📧 nikita.porwal05@gmail.com
🔗 https://linkedin.com/in/nikita-porwal

## 📊 Overview
This project predicts **property sale prices** using historical real estate data.  

It demonstrates a complete **data science workflow**, including:
- Data cleaning  
- Feature engineering  
- Handling missing values  
- Preparing data for regression modelling  

---

## Problem Statement
Real estate pricing depends on multiple factors such as location, property size, facilities, and infrastructure.

The objective is to:
- Build a model that predicts property prices  
- Understand factors influencing prices  
- Prepare clean and structured data for regression  

---

## Project Workflow

### 🔹 Data Preprocessing
- Loaded training and test datasets  
- Combined datasets for consistent cleaning  
- Identified missing values and inconsistencies  
- Removed unnecessary columns  

---

### 🔹 Handling Missing Values

Applied domain-specific imputation strategies:

- Filled categorical missing values with meaningful labels:
  - "No Pool", "No Garage", "No Basement"  
- Imputed numerical values using:
  - Median (Lot_Extent)  
- Mode imputation for categorical features  

---

### 🔹 Feature Engineering

- Created consistent dataset structure  
- Handled garage and basement related features  
- Removed low-importance features (e.g., Utility Type)  

---

### 🔹 Data Cleaning Techniques Used

- Missing value analysis  
- Column removal  
- Mode / Median imputation  
- Data transformation and preparation  

---

## Model

- Applied **Linear Regression**  
- Ensured compliance with regression assumptions:
  - Linearity  
  - No multicollinearity  
  - Homoscedasticity  

---

## Key Highlights

- Strong focus on **data preparation and cleaning**  
- Applied **domain knowledge for feature imputation**  
- Built structured dataset ready for regression modelling  

---

## Tech Stack

- **Language:** Python  
- **Libraries:**
  - pandas, numpy  
  - matplotlib, seaborn  
  - scikit-learn  

