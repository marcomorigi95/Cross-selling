# Cross Selling
---

## Run in Google Colab

Run the full project in Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1yENVigYwRaR4YecU-3WjbkzmBEKtylQD?usp=sharing)

---

# Health Insurance Cross-Sell Prediction

**Client:** A leading Indian insurance company
**Objective:** Predict whether current health insurance policyholders are likely to purchase **vehicle insurance**, using demographic, vehicle, and policy data.

---

## Business Context

The client, based in **India**, provides health insurance and wants to expand cross-selling to **vehicle insurance**. The goal is to develop a **machine learning model** that predicts customer interest in vehicle insurance based on historical data, improving targeting and reducing random marketing efforts.

**Dataset Source:**
[Health Insurance Cross-Sell Prediction – Kaggle](https://www.kaggle.com/datasets/anmolkumar/health-insurance-cross-sell-prediction/data)

---

## Data Overview

The dataset includes information in the following categories:

* **Demographics:**
  `Gender`, `Age`, `Region_Code`, `Previously_Insured`

* **Vehicle Details:**
  `Vehicle_Age`, `Vehicle_Damage`

* **Policy Details:**
  `Annual_Premium`, `Policy_Sales_Channel`, `Vintage`

* **Target Variable:**
  `Response` — Whether the customer showed interest in vehicle insurance

**Note:**
Given the Indian context, **currency values**, **demographic distributions**, and **vehicle ownership trends** are interpreted accordingly.

---

## Project Workflow

### 1. Exploratory Data Analysis (EDA)

* Univariate & bivariate analysis of categorical and numerical features
* Correlation analysis using `dython` for categorical features
* Statistical distribution comparisons using **Kolmogorov-Smirnov test**

### 2. Preprocessing & Feature Engineering

* Encoding:

  * `OneHotEncoder` for nominal features
  * `OrdinalEncoder` for ordered categories
* Scaling with `StandardScaler`
* Handling class imbalance using:

  * `RandomOverSampler`, `RandomUnderSampler`
  * `SMOTENC` (for categorical + numerical data)

### 3. Model Development

* Base model: `LogisticRegression`
* Hyperparameter tuning with `RandomizedSearchCV`
* Training with `train_test_split` (stratified)
* Pipeline construction with `Pipeline` and `ColumnTransformer`

### 4. Evaluation Metrics

* **F1 Score**
* **ROC-AUC**
* **Recall / Precision**
* **Confusion Matrix**
* **Classification Report**
* Visuals: `RocCurveDisplay`, `PrecisionRecallDisplay`

### 5. Model Interpretation

* Local interpretability with **LIME** (`lime.lime_tabular`)
* Feature impact and decision insights for individual predictions

