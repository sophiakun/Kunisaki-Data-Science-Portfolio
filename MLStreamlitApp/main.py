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
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler, LabelEncoder

# -----------------------------------------------
# Application Information
# -----------------------------------------------
st.set_page_config(page_title="Iris Classifier", layout="wide")
st.title("Iris Classifier") 
st.markdown("""
### Interactive Machine Learning  Explorer
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
# Load and Prepare Data
# -----------------------------------------------
df, X, y, features = load_and_preprocess_data()

# Encode target for classification
le = LabelEncoder()
y_class = le.fit_transform(y)

# For regression, we'll use the first feature as target
y_reg = X.iloc[:, 0]  # Using sepal_length as regression target
X_reg = X.iloc[:, 1:]  # Other features as predictors

# Split data for classification
X_train_class, X_test_class, y_train_class, y_test_class = split_data(X, y_class)

# Split data for regression
X_train_reg, X_test_reg, y_train_reg, y_test_reg = split_data(X_reg, y_reg)

# -----------------------------------------------
# Select Type of ML Model 
# -----------------------------------------------
st.sidebar.header("Model Configuration")
model_type = st.sidebar.radio(
    "Select Algorithm:",
    ["Linear Regression", "K-Nearest Neighbors", "Decision Tree"],  
)

# -----------------------------------------------
# Linear Regression
# -----------------------------------------------
if model_type == "Linear Regression":
    st.header("Linear Regression")
    model = LinearRegression()
    model.fit(X_train_reg, y_train_reg)
    y_pred = model.predict(X_test_reg)

    st.write(f"**R² Score:** {r2_score(y_test_reg, y_pred):.2f}")
    st.write(f"**Mean Squared Error:** {mean_squared_error(y_test_reg, y_pred):.2f}")

    st.subheader("Predicted vs Actual")
    fig, ax = plt.subplots()
    ax.scatter(y_test_reg, y_pred)
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title("Linear Regression: Actual vs. Predicted")
    st.pyplot(fig)

