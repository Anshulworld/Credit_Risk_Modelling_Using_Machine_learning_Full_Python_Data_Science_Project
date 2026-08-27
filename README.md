# Credit Risk Modelling Using Machine Learning 📊

A full-stack Data Science and Machine Learning project designed to predict credit risk. This project analyzes financial data to determine the likelihood of a customer defaulting on a loan, helping financial institutions make informed, data-driven lending decisions.

## 📝 Project Overview

This repository contains the complete pipeline for a credit risk modeling project, from Exploratory Data Analysis (EDA) and data preprocessing to model training and deployment. 

Based on the `german_credit_data.csv`, the project uses various machine learning techniques, culminating in a highly accurate **Extra Trees Classifier** model. The project also includes a user-friendly web application (`app.py`) for making real-time predictions.

## 🗂 Repository Structure

*   `Analysis_model.ipynb`: The main Jupyter Notebook containing data exploration, preprocessing, model training, and evaluation.
*   `app.py`: The web application script (likely Streamlit or Flask) to deploy the model and interact with it via a UI.
*   `german_credit_data.csv`: The dataset used for training and testing.
*   `extra_trees_credit_model.pkl`: The saved Extra Trees Machine Learning model.
*   **Encoders**: Saved `.pkl` files used to transform categorical data into numerical format for the model:
    *   `Checking account_encoder.pkl`
    *   `Housing_encoder.pkl`
    *   `Saving accounts_encoder.pkl`
    *   `Sex_encoder.pkl`
    *   `target_encoder.pkl`
*   `Screenshots/`: Directory containing all visual analysis charts.

## 📈 Exploratory Data Analysis (EDA) & Visualizations

During the data analysis phase, several visualizations were created to understand the distribution of features and their relationship with credit risk. Below are the key insights and charts generated:

### Distribution of Numerical Features
Understanding how our continuous variables are spread across the dataset.
![Distribution of Numerical Features](Screenshots/Distribution%20of%20Numerical%20Features.png)

### Correlation Heatmap
Identifying the relationships and potential multicollinearity between different numerical features.
![Heatmap](Screenshots/Heatmap.png)

### Scatterplot Analysis
Visualizing the direct relationship between specific pairs of variables.
![Scatterplot](Screenshots/Scatterplot.png)

### Subplots & Countplots
Breaking down categorical variables to see the frequency of different classes (e.g., how many risky vs. good loans exist per category).
![Subplot and Countplot](Screenshots/Subplot%20and%20Countplot.png)

### Barchart
![Barchart](Screenshots/Barchart%20.png)

### Boxplots (Outlier Detection)
Analyzing the spread of the data and identifying potential outliers in features like credit amount or age.
![BoxPlot](Screenshots/BoxPlot.png)
![Boxplot and Subplot](Screenshots/boxplot%20and%20subplot.png)

### Violin Plots
Combining the benefits of boxplots and KDE plots to see the exact distribution shape and density across categories.
![Violinplot](Screenshots/Violinplot.png)

## 🚀 How to Run the Project Locally

**1. Clone the repository:**
```bash
git clone [https://github.com/Anshulworld/Credit_Risk_Modelling_Using_Machine_learning_Full_Python_Data_Science_Project.git](https://github.com/Anshulworld/Credit_Risk_Modelling_Using_Machine_learning_Full_Python_Data_Science_Project.git)
cd Credit_Risk_Modelling_Using_Machine_learning_Full_Python_Data_Science_Project
