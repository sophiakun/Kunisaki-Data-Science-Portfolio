# -----------------------------------------------
# Preliminary Steps
# -----------------------------------------------

# # requirements.txt file commands:
# pip install pipreqs

# import libraries 
import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler, LabelEncoder
    
# -----------------------------------------------
# Application Information
# -----------------------------------------------

st.title("Supervised Machine Learning Explorer")
st.markdown("""
Explore different supervised machine learning models including linear and logistic regression, decision trees,
or K-nearest neighrbors with this interactive app. Upload your own dataset or use the built-in Titanic and Iris datasets.
""")

with st.sidebar:
    st.header("Dataset Selection")
    dataset_choice = st.selectbox(
        "Choose a dataset",
        ["Titanic (Classification)", "Iris (Classification)", "Upload your own"]
    )

    st.header("Model Settings")
    model_type = st.selectbox(
        "Select model type",
        ["Logistic Regression", "Decision Tree", "K-Nearest Neighbors", "Linear Regression"]
    )

# -----------------------------------------------
# Helper Functions
# -----------------------------------------------
@st.cache_data
def load_data(dataset_name):
    if dataset_name == "Titanic (Classification)":
        df = sns.load_dataset('titanic')
        df = df.dropna(subset=['age'])
        df = pd.get_dummies(df, columns=['sex'], drop_first=True)
        features = ['pclass', 'age', 'sibsp', 'parch', 'fare', 'sex_male']
        X = df[features]
        y = df['survived']
        return X, y, features, df
        
    elif dataset_name == "Iris (Classification)":
        iris = load_iris()
        X = pd.DataFrame(iris.data, columns=iris.feature_names)
        y = pd.Series(iris.target)
        features = iris.feature_names
        df = pd.concat([X, y], axis=1)
        df.columns = list(iris.feature_names) + ['target']
        return X, y, features, df
   
    else:  # Upload your own
        uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
        return None, None, None, None


# -----------------------------------------------
# Model Selection and Configuration
# -----------------------------------------------


# -----------------------------------------------
# Linear Regression
# -----------------------------------------------

# -----------------------------------------------
# Decision Tree
# -----------------------------------------------


# -----------------------------------------------
# Dataset Info for All Tabs
# -----------------------------------------------
