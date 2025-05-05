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
st.sidebar.header("2. Select Features for Clustering")

# Get numeric columns (from the current df!)
numeric_cols = df.select_dtypes(include="number").columns.tolist()

# Set prompt + default features smartly
if source == "Iris Dataset":
    prompt = "Select flower measurements to include in clustering:"
    default_features = numeric_cols  # All 4 iris columns

elif source == "Titanic Dataset":
    prompt = "Select passenger features to include in clustering:"
    default_features = [col for col in ['pclass', 'age', 'sibsp', 'parch', 'fare', 'sex_male'] if col in numeric_cols]

else:
    prompt = "Select numeric columns to include in clustering:"
    default_features = numeric_cols

# Handle case: no numeric columns at all
if not numeric_cols:
    st.warning("⚠️ No numeric columns available for clustering.")
    st.stop()

# Add unique key for Streamlit widget reset
feature_cols = st.sidebar.multiselect(
    prompt,
    options=numeric_cols,
    default=default_features,
    key=f"feature_select_{source}"
)

X = df[feature_cols]
st.sidebar.write(f"Selected features: {', '.join(X.columns)}")
