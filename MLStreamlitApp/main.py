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
# Select Type of ML Model & Scale
# -----------------------------------------------
st.sidebar.header("Model Configuration")
model_type = st.sidebar.radio(
    "Select Algorithm:",
    ["Linear Regression", "K-Nearest Neighbors", "Decision Tree"],  
)

# -----------------------------------------------
# Model-Specific Logic
# -----------------------------------------------

if model_type == "Linear Regression":
    st.header("Linear Regression")
    model = LinearRegression()
    model.fit(X_train, y_train_reg)
    y_pred = model.predict(X_test)

    st.write(f"**R² Score:** {r2_score(y_test_reg, y_pred):.2f}")
    st.write(f"**Mean Squared Error:** {mean_squared_error(y_test_reg, y_pred):.2f}")

    st.subheader("Predicted vs Actual")
    fig, ax = plt.subplots()
    ax.scatter(y_test_reg, y_pred)
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title("Linear Regression: Actual vs. Predicted")
    st.pyplot(fig)

elif model_type == "KNN Classifier":
    st.header("K-Nearest Neighbors (KNN)")
    k = st.slider("Select number of neighbors (k)", 1, 15, step=2, value=5)
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train_class)
    y_pred = knn.predict(X_test)

    st.write(f"**Accuracy:** {accuracy_score(y_test_class, y_pred):.2f}")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test_class, y_pred)
        plot_confusion_matrix(cm, "KNN Confusion Matrix")

    with col2:
        st.subheader("Classification Report")
        st.text(classification_report(y_test_class, y_pred))

elif model_type == "Decision Tree":
    st.header("Decision Tree Classifier")
    max_depth = st.slider("Max Depth of Tree", 1, 10, step=1, value=3)
    dtree = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    dtree.fit(X_train, y_train_class)
    y_pred = dtree.predict(X_test)

    st.write(f"**Accuracy:** {accuracy_score(y_test_class, y_pred):.2f}")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test_class, y_pred)
        plot_confusion_matrix(cm, "Decision Tree Confusion Matrix")

    with col2:
        st.subheader("Classification Report")
        st.text(classification_report(y_test_class, y_pred))

# -----------------------------------------------
# Dataset Info
# -----------------------------------------------
with st.expander("Click to view Iris Dataset Information"):
    st.write("### First 5 Rows of the Dataset")
    st.dataframe(df.head())
    st.write("### Summary Statistics")
    st.dataframe(df.describe())