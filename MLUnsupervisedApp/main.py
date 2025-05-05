# -----------------------------------------------
# Launch Streamlit Cloud App
# -----------------------------------------------

# requirements.txt file commands: pip install pipreqs

# -----------------------------------------------
# Unsupervised Machine Learning Streamlit App
# -----------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris

# -----------------------------------------------
# Application Information
# -----------------------------------------------

st.title("⚙️  Unsupervised Machine Learning Explorer ⚙️ ")
st.markdown("""
Explore different unsupervised machine learning models including:  
1. **K-Means Clustering**: an algorithm that partitions data into distinct clusters  
   by minimizing within-cluster variance.  
2. **Principal Component Analysis (PCA)**: a dimensionality reduction technique that  
   transforms features into principal components capturing the most variance.

Upload **your own dataset** or use the built-in **Iris Dataset** to engage with this interactive app!
""")

# -----------------------------------------------
# User-Uploaded Dataset
# -----------------------------------------------

# Create sidebar with different datasets to choose from
st.sidebar.header("1. Choose a Dataset")
source = st.sidebar.radio("Select dataset source:", ["Iris Dataset", "Titanic Dataset", "Upload your own CSV"])

# for user-uploaded CSV
if source == "Upload your own CSV":
    uploaded_file = st.sidebar.file_uploader("Upload CSV", type="csv")
    # If no file is uploaded, show a warning and stop the app from continuing
    if not uploaded_file:
        st.warning("Please upload a CSV file to continue.")
        st.stop()
    df = pd.read_csv(uploaded_file)
# for the Titanic example
else:
    if source == "Titanic":
        # Drop rows with missing age values
        df = sns.load_dataset("titanic").dropna(subset=["age"])
        # Convert the sex column to a binary numeric column
        df = pd.get_dummies(df, columns=["sex"], drop_first=True)
        # Define features and target
        features = ['pclass', 'age', 'sibsp', 'parch', 'fare', 'sex_male']
        X = df[features]
        y = df["survived"]
    else: # for the Iris example
        df = sns.load_dataset("iris")
        # Use all columns except species as features
        X = df.drop(columns=["species"])
        # Make species as target
        y = df["species"]

# -----------------------------------------------
# Feature Selection Sidebar
# -----------------------------------------------

if source == "Upload your own CSV":
    st.sidebar.header("2. Select Features for Clustering")

    # Let the user pick which columns to use
    feature_cols = st.sidebar.multiselect(
        "Select features to include in clustering:",
        options=df.columns,
        default=df.select_dtypes(include="number").columns.tolist()  # default = numeric columns
    )

    # Filter to numeric columns only (in case user selects non-numeric)
    X = df[feature_cols].select_dtypes(include="number")

    # Display what features were selected
    st.write(f"Selected features: {', '.join(X.columns)}")

else:
    # Predefine features for built-in datasets

    if source == "Titanic Dataset":
        st.sidebar.header("2. Clustering Features")
        st.sidebar.write("""
        Using the following features for clustering:
        - pclass
        - age
        - sibsp
        - parch
        - fare
        - sex_male
        """)
        # Make sure only columns that exist are selected (in case)
        feature_cols = [col for col in ['pclass', 'age', 'sibsp', 'parch', 'fare', 'sex_male'] if col in df.columns]
        X = df[feature_cols]

    elif source == "Iris Dataset":
        st.sidebar.header("2. Clustering Features")
        st.sidebar.write("""
        Using all 4 iris flower measurements:
        - sepal length (cm)
        - sepal width (cm)
        - petal length (cm)
        - petal width (cm)
        """)
        # Just the numeric columns (already known to be 4 in Iris dataset)
        feature_cols = df.select_dtypes(include="number").columns.tolist()
        X = df[feature_cols]

