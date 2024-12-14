# Taxi Trip Data Analysis and Machine Learning Models

This repository contains a comprehensive workflow for analyzing, processing, and building machine learning models on the **2021 NYC Green Taxi Trip Dataset**. The code includes advanced data preprocessing techniques, feature selection methods, and predictive modeling approaches to analyze taxi trip data and derive insights.

---

## Features

### 1. **Data Preprocessing**
   - Loading the NYC Green Taxi dataset and related borough information.
   - Cleaning data by handling missing values using regression and classification methods.
   - Transforming date and time fields to extract useful features (e.g., hour, day, and weekday).
   - Encoding categorical variables such as boroughs using one-hot encoding.
   - Dropping irrelevant or redundant columns to streamline the dataset.

### 2. **Exploratory Data Analysis (EDA)**
   - Visualizing activity levels using a **heatmap** of trips by hour and weekday.
   - Analyzing trip type and payment type distributions by hour using bar plots.
   - Assessing green taxi usage trends across boroughs and hours.

### 3. **Feature Engineering and Selection**
   - **Backward Elimination**: Removing features iteratively to optimize model performance.
   - **Forward Selection**: Adding features incrementally to find the most predictive variables.
   - **Random Forest Feature Importance**: Using Random Forest to select features with the highest importance scores.
   - **Chi-Square Test**: Selecting features based on statistical relevance.

### 4. **Predictive Modeling**
   - **Classification Models**:
     - Decision Tree
     - Random Forest
     - XGBoost
     - Evaluating models using accuracy and classification reports for predicting tipping behavior.
   - **Regression Models**:
     - Linear Regression
     - Random Forest Regressor
     - XGBoost Regressor
     - Comparing models using metrics like Mean Squared Error (MSE) and R² score.

### 5. **Model Comparison**
   - A detailed comparison of regression models for predicting the total fare amount.
   - Identifying the best-performing model using evaluation metrics.

---

## Visualizations
- Heatmaps for activity levels by time and day.
- Bar plots for trip and payment type distributions.
- Correlation matrix for feature relationships.
- Borough-specific trends in green taxi usage.


---

## Dataset
- **2021 NYC Green Taxi Trip Data**: Public dataset available from NYC Taxi & Limousine Commission (TLC).
- **Borough Mapping Data**: Mapping file for borough names and location IDs.

---

## Dependencies
- Python 3.8+
- pandas, numpy, matplotlib, seaborn
- scikit-learn, xgboost

---

## Notice
For the latest and most detailed implementation with explanations, please refer to the Jupyter Notebook file included in the repository. 

---

