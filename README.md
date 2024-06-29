# Prasunethon-Hackathon

# Credit Card Fraud Detection

This project involves building and evaluating various machine learning models to detect fraudulent transactions in a credit card dataset.

-Dataset
The dataset used in this project is creditcard.csv, which contains transactions made by credit cards in September 2013 by European cardholders. The dataset presents transactions that occurred in two days, with 492 frauds out of 284,807 transactions.

# Project Structure
1. Data Exploration

-Load the dataset.
-Display basic statistics and data structure.
-Visualize class distribution.
-Analyze fraudulent and non-fraudulent transactions.
-Visualize the distribution of transaction time and amount.
-Correlation heatmap.

2. Data Preprocessing

-Standardize the features.
-Split the data into training, validation, and testing sets.

3. Model Training and Evaluation

-Train multiple models:
-Artificial Neural Networks (ANNs)
-XGBoost
-Random Forest
-CatBoost
-LightGBM
-Evaluate models using metrics like accuracy, precision, recall, and F1 score.
-Compare model performance.

# Installation
Ensure you have Python and the required libraries installed:
pip install pandas numpy matplotlib seaborn scikit-learn xgboost catboost lightgbm

# Usage
1. Load and Explore the Data:
   import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("creditcard.csv")

2. Preprocess the Data:
   from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

scalar = StandardScaler()
X = data.drop('Class', axis=1)
y = data.Class
X_train_v, X_test, y_train_v, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
X_train, X_validate, y_train, y_validate = train_test_split(X_train_v, y_train_v, test_size=0.2, random_state=42)
X_train = scalar.fit_transform(X_train)
X_validate = scalar.transform(X_validate)
X_test = scalar.transform(X_test)

3. Train and Evaluate Models:
   from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score

def print_score(label, prediction, train=True):
    label = label.astype(int)
    prediction = prediction.astype(int)
    clf_report = pd.DataFrame(classification_report(label, prediction, output_dict=True))
    print("Train Result:" if train else "Test Result:")
    print(f"Accuracy Score: {accuracy_score(label, prediction) * 100:.2f}%")
    print(f"Classification Report:\n{clf_report}")
    print(f"Confusion Matrix: \n {confusion_matrix(label, prediction)}\n")

# Example with XGBoost
from xgboost import XGBClassifier
xgb_clf = XGBClassifier()
xgb_clf.fit(X_train, y_train)
y_train_pred = xgb_clf.predict(X_train)
y_test_pred = xgb_clf.predict(X_test)
print_score(y_train, y_train_pred, train=True)
print_score(y_test, y_test_pred, train=False)

4. Plot Model Performance:
   import matplotlib.pyplot as plt
scores_dict = {
    'XGBoost': {
        'Train': f1_score(y_train, y_train_pred),
        'Test': f1_score(y_test, y_test_pred),
    },
}
scores_df = pd.DataFrame(scores_dict)
scores_df.plot(kind='barh', figsize=(15, 8))

# Results
-Detailed analysis and visualization of the dataset.
-Comparison of multiple machine learning models.
-Insights into model performance and selection of the best-performing model.

# Contributing
If you have suggestions or improvements, feel free to open an issue or submit a pull request.

# Acknowledgments
-The dataset is provided by Kaggle.
-Libraries used: Pandas, NumPy, Matplotlib, Seaborn, Scikit-Learn, XGBoost, CatBoost, LightGBM.


