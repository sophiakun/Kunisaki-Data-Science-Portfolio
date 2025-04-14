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

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler

# -----------------------------------------------
# Application Information
# -----------------------------------------------
st.set_page_config(page_title="Iris Classifier", layout="wide")
st.title("Iris Classifier")  # Updated title
st.markdown("""
### Interactive Classification Explorer
### Compare Classification Algorithms
1. **Linear Regression**: Continuous prediction 
2. **K-Nearest Neighbors (KNN)**: Distance-based
3. **Decision Tree**: Rule-based
""")

# -----------------------------------------------
# Helper Functions
# -----------------------------------------------
def load_and_preprocess_data():
    df = pd.read_csv("iris.csv")
    features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    X = df[features]
    y = df['species'] 
    return df, X, y, features

def split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def train_knn(X_train, y_train, n_neighbors):
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train, y_train)
    return knn

def plot_confusion_matrix(cm, classes, title):
    plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    st.pyplot(plt)
    plt.clf()

# -----------------------------------------------
# Select Type of ML Model
# -----------------------------------------------
st.sidebar.header("Model Configuration")
algorithm = st.sidebar.radio(
    "Select Algorithm",
    ["Linear Regression", "K-Nearest Neighbors", "Decision Tree"],  
    index=0  
)
