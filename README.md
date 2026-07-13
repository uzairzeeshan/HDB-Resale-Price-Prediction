# 🇸🇬 Singapore HDB Resale Price Prediction

## Overview

The **Singapore HDB Resale Price Prediction** project is a machine learning application developed to predict the resale prices of Housing and Development Board (HDB) flats in Singapore based on various property attributes. The objective of this project is to build an accurate predictive model that assists buyers, sellers, and real estate professionals in estimating property prices using historical housing data. By leveraging data preprocessing, exploratory data analysis (EDA), feature engineering, and machine learning techniques, the project demonstrates how predictive analytics can support data-driven decision-making in the real estate sector.

## Dataset

The project utilizes historical Singapore HDB resale transaction data containing thousands of property records. The dataset includes important attributes such as:

* Town
* Flat Type
* Flat Model
* Floor Area
* Storey Range
* Lease Commencement Date
* Remaining Lease
* Resale Month
* Property Age
* Resale Price (Target Variable)

Multiple datasets are combined and transformed to create a clean and structured dataset suitable for machine learning.

## Project Workflow

The project follows a complete end-to-end machine learning pipeline:

* Imported essential Python libraries including **NumPy**, **Pandas**, **Matplotlib**, **Seaborn**, and **Scikit-learn**.
* Loaded and merged multiple HDB datasets.
* Performed data cleaning by handling missing values, removing duplicates, and correcting inconsistent data.
* Conducted **Exploratory Data Analysis (EDA)** to understand price trends, feature distributions, and correlations.
* Applied **Feature Engineering** by extracting meaningful variables such as property age and remaining lease.
* Encoded categorical variables using suitable encoding techniques.
* Split the dataset into training and testing datasets.
* Trained multiple regression models and compared their performance.
* Saved the best-performing model using **Pickle** for future predictions.
* Developed a **Streamlit** web application that allows users to enter property details and instantly predict resale prices.

## Machine Learning Models

Several regression algorithms were evaluated, including:

* Linear Regression
* Decision Tree Regressor
* Random Forest Regressor

After comparing model performance, the **Decision Tree Regressor** was selected as the final model because it provided the most accurate resale price predictions for the dataset.

## Model Evaluation

The regression models were evaluated using standard performance metrics such as:

* R² Score
* Mean Absolute Error (MAE)
* Mean Squared Error (MSE)
* Root Mean Squared Error (RMSE)

These metrics were used to compare prediction accuracy and select the most reliable model.

## Technologies Used

* Python
* NumPy
* Pandas
* Matplotlib
* Seaborn
* Scikit-learn
* Pickle
* Streamlit
* Jupyter Notebook / Google Colab

## Key Features

* End-to-end machine learning workflow
* Comprehensive data preprocessing and feature engineering
* Exploratory Data Analysis with visualizations
* Multiple regression model comparison
* Model serialization using Pickle
* Interactive Streamlit web application
* Real-time resale price prediction

## Future Enhancements

Future improvements include implementing advanced ensemble algorithms such as **XGBoost**, **LightGBM**, and **CatBoost** to improve prediction accuracy. The project can also integrate location-based features, nearby amenities, MRT station distances, and geospatial information. Deploying the application on cloud platforms such as **Render**, **Railway**, or **Streamlit Community Cloud** would enable users to access the prediction system online.

## Conclusion

The Singapore HDB Resale Price Prediction project demonstrates how machine learning can be applied to solve real-world real estate problems by accurately estimating housing prices. Through effective data preprocessing, feature engineering, model training, and deployment with Streamlit, the project provides a practical solution for predicting property values. It showcases essential data science skills, including data analysis, predictive modeling, model evaluation, and web application deployment, making it an excellent portfolio project for aspiring Data Analysts and Machine Learning Engineers.
