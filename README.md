# 🏦 Credit Risk Modelling Using Machine Learning

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-orange.svg)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-red.svg)](https://streamlit.io/)

## 📌 Project Overview

In the highly regulated financial sector, accurately assessing the creditworthiness of applicants is critical for minimizing non-performing loans (NPLs) and maximizing institutional profitability. 

This project is a complete, end-to-end **Data Science and Machine Learning solution** designed to predict consumer credit risk. By analyzing historical financial data and demographic attributes, the model classifies prospective loan applicants as either a **"Good"** or **"Bad"** credit risk. This automated predictive pipeline assists banks and financial institutions in making faster, data-driven, and objective lending decisions.

## 🚀 The Machine Learning Lifecycle & Methodology

This repository showcases a full-stack data science workflow, from raw data processing to web-based model inference.

### 1. Exploratory Data Analysis (EDA) & Preprocessing
* **Data Profiling:** Analyzed the German Credit Dataset to understand the distribution of numerical (e.g., credit amount, duration) and categorical (e.g., housing, purpose) variables.
* **Feature Engineering:** Handled categorical data using custom encoders to transform string inputs into machine-readable formats without losing contextual weight. These transformations are serialized as `.pkl` files to ensure strict consistency between the training environment and production.

### 2. Predictive Modeling (Extra Trees Classifier)
Instead of a standard Decision Tree or Random Forest, this project implements an **Extra Trees (Extremely Randomized Trees) Classifier**. 
* **Why Extra Trees?** Financial datasets often suffer from high variance and complex dimensional relationships. Extra Trees introduces a higher level of randomization by selecting cut-points at random rather than seeking the optimum split. This significantly **reduces model variance and mitigates overfitting**, making it highly robust for credit scoring.

### 3. Production Deployment
* **Interactive UI:** The model is deployed via a real-time web application (`app.py`). Loan officers or stakeholders can input applicant details through an intuitive interface and instantly receive a computed credit risk classification.

## 📊 Dataset Details

This project utilizes the well-known **German Credit Dataset** (`german_credit_data.csv`). It contains granular attributes of loan applicants, broadly categorized into:
* **Financial Standing:** Checking account status, savings account balance.
* **Personal Demographics:** Age, sex, and housing situation (rent, own, free).
* **Loan Characteristics:** Duration of the loan (in months), requested credit amount, and the specific purpose of the loan (e.g., car, education, furniture).

*Note: Further metadata and column descriptions can be found in `Data.txt`.*

## 🛠️ Tech Stack & Tools

* **Programming Language:** Python
* **Machine Learning:** Scikit-Learn (Extra Trees Classifier)
* **Data Manipulation:** Pandas, NumPy
* **Model Serialization:** Pickle (for saving pre-trained models and encoders)
* **Web Deployment:** Streamlit (via `app.py`)
* **Development Environment:** Jupyter Notebook

## 📂 Repository Architecture

```text
├── Analysis_model.ipynb           # Comprehensive Notebook: EDA, preprocessing, and model training
├── app.py                         # Streamlit web application script for real-time inference
├── german_credit_data.csv         # The core dataset used for training and testing
├── Data.txt                       # Metadata and data dictionary
├── extra_trees_credit_model.pkl   # The serialized, pre-trained Extra Trees predictive model
├── target_encoder.pkl             # Serialized encoder for the target variable (Good/Bad)
├── Checking account_encoder.pkl   # Serialized categorical encoder for Checking accounts
├── Housing_encoder.pkl            # Serialized categorical encoder for Housing status
├── Saving accounts_encoder.pkl    # Serialized categorical encoder for Savings accounts
└── Sex_encoder.pkl                # Serialized categorical encoder for Applicant Sex

```

⚙️ How to Run the Application Locally
Follow these steps to deploy the credit risk application on your local machine:

1. Clone the Repository

git clone [https://github.com/Anshulworld/Credit_Risk_Modelling_Using_Machine_learning_Full_Python_Data_Science_Project.git](https://github.com/Anshulworld/Credit_Risk_Modelling_Using_Machine_learning_Full_Python_Data_Science_Project.git)
cd Credit_Risk_Modelling_Using_Machine_learning_Full_Python_Data_Science_Project

2.Install Dependencies
Ensure you have Python installed. Install the required libraries using pip:
pip install pandas numpy scikit-learn streamlit

3. Launch the Web Application
Run the Streamlit server: streamlit run app.py

4. Access the App
Open your web browser and navigate to: http://localhost:8501/


👨‍💻 Author
Anshul Kumar Singh

Data Science & Analytics Enthusiast

Passionate about building end-to-end analytical solutions, predictive models, and intuitive data applications.



