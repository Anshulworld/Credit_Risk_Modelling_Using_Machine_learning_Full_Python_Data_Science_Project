# Credit_Risk_Modelling_Using_Machine_learning_Full_Python_Data_Science_Project

# 🏦 Credit Risk Modelling Using Machine Learning

## 📌 Project Overview
In the financial sector, assessing the creditworthiness of applicants is critical for minimizing default risks and maximizing profitability. This project is an end-to-end **Data Science and Machine Learning solution** designed to predict credit risk. Using historical financial data, the model classifies loan applicants as either a "Good" or "Bad" credit risk. 

By automating the initial screening process, this tool assists financial institutions in making faster, data-driven lending decisions.

## 🚀 Key Features & Highlights
* **End-to-End Pipeline:** Showcases a complete machine learning lifecycle, from Exploratory Data Analysis (EDA) to model deployment.
* **Robust Feature Engineering:** Utilizes custom categorical encoders (saved as `.pkl` files) to seamlessly handle real-world string inputs for variables like housing, sex, and checking/savings accounts.
* **Advanced Machine Learning:** Implements an **Extra Trees Classifier**, chosen for its efficiency and ability to reduce variance in highly dimensional financial data.
* **Interactive Web Application:** Includes an `app.py` script that serves as a user interface, allowing stakeholders to input applicant details and receive real-time credit risk predictions.

## 🛠️ Tech Stack & Tools
* **Programming Language:** Python
* **Machine Learning:** Scikit-Learn (Extra Trees Classifier)
* **Data Processing:** Pandas, NumPy
* **Deployment/UI:** Streamlit / Flask (via `app.py`)
* **Environment:** Jupyter Notebook

## 📂 Repository Structure
```text
├── Analysis_model.ipynb           # Complete Jupyter Notebook with EDA, preprocessing, and model training
├── app.py                         # Web application script for real-time model inference
├── german_credit_data.csv         # The dataset used for training and testing
├── Data.txt                       # Metadata and column descriptions for the dataset
├── extra_trees_credit_model.pkl   # The serialized, pre-trained Extra Trees model
├── target_encoder.pkl             # Serialized encoder for the target variable
├── Checking account_encoder.pkl   # Encoder for Checking account feature
├── Housing_encoder.pkl            # Encoder for Housing feature
├── Saving accounts_encoder.pkl    # Encoder for Saving accounts feature
└── Sex_encoder.pkl                # Encoder for Sex feature
```

## Dataset
The project utilizes the German Credit Dataset (german_credit_data.csv). It contains various attributes of loan applicants, including:

Financial Standing: Checking and savings account status

Personal Demographics: Age, sex, housing situation

Loan Characteristics: Duration, credit amount, and purpose

## How to Run the Application Locally

1. Clone the Repository
git clone [https://github.com/Anshulworld/Credit_Risk_Modelling_Using_Machine_learning_Full_Python_Data_Science_Project.git](https://github.com/Anshulworld/Credit_Risk_Modelling_Using_Machine_learning_Full_Python_Data_Science_Project.git)
cd Credit_Risk_Modelling_Using_Machine_learning_Full_Python_Data_Science_Project

2. Install necessary dependencies:
Ensure you have Python installed, then install the required libraries.
pip install pandas numpy scikit-learn streamlit

3. Run the web application:
streamlit run app.py


# Author
## Anshul Kumar Singh
# Data Science & Analytics Enthusiast
