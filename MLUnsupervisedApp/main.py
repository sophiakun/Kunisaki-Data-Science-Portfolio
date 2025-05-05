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

# Load dataset based on user selection
if source == "Upload your own CSV":
    uploaded_file = st.sidebar.file_uploader("Upload CSV", type="csv")
    # If no file is uploaded, show a warning and stop the app from continuing
    if not uploaded_file:
        st.warning("Please upload a CSV file to continue.")
        st.stop()
    df = pd.read_csv(uploaded_file)

elif source == "Titanic Dataset":
    df = sns.load_dataset("titanic").dropna(subset=["age"])
    df = pd.get_dummies(df, columns=["sex"], drop_first=True)

else:  # Iris Dataset
    df = sns.load_dataset("iris")

# -----------------------------------------------
# Feature Selection Sidebar
# -----------------------------------------------

# Create header in sidebar for feature selection
st.sidebar.header("2. Select Features for Clustering")

# Get numeric columns 
numeric_cols = df.select_dtypes(include="number").columns.tolist()

# Set prompt and give default features
if source == "Iris Dataset":
    prompt = "Select flower measurements to include in clustering:"
    default_features = numeric_cols  # All 4 iris columns

elif source == "Titanic Dataset":
    prompt = "Select passenger features to include in clustering:"
    default_features = [col for col in ['pclass', 'age', 'sibsp', 'parch', 'fare', 'sex_male'] if col in numeric_cols]

else:
    prompt = "Select numeric columns to include in clustering:"
    default_features = numeric_cols

# Restrict to numeric columns 
if not numeric_cols:
    st.warning("No numeric columns available for clustering.")
    st.stop()

# Select features
feature_cols = st.sidebar.multiselect(
    prompt,
    options=numeric_cols,
    default=default_features)

X = df[feature_cols]
# Display the list of selected feature names to the user
st.sidebar.write(f"Selected features: {', '.join(X.columns)}")

# -----------------------------------------------
# Model Selection Sidebar
# -----------------------------------------------

# Create header in sidebar for model selection
st.sidebar.header("3. Choose a Model")
model_choice = st.sidebar.selectbox("Model:", ["K-Means Clustering", "Principal Component Analysis (PCA)"])

# K-Means settings
if model_choice == "K-Means Clustering":
    k = st.sidebar.slider("Number of clusters (k):", 2, 10, 3)
    init_method = st.sidebar.selectbox("Initialization method:", ["k-means++", "random"])

# PCA settings
if model_choice == "Principal Component Analysis (PCA)":
    max_components = min(len(feature_cols), 10)
    n_components = st.sidebar.slider(
        "Number of components:", min_value=2, max_value=max_components, value=2)

# -----------------------------------------------
# Create 3 Different Tabs 
# -----------------------------------------------

# Create tabs: About / Model Settings / Evaluation
tab1, tab2, tab3 = st.tabs(["About", "Model Settings", "Evaluation"])

# Tab 1 
with tab1:
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    if source == "Titanic Dataset":
        st.markdown("""
        **Titanic Dataset Overview:**
        - This dataset includes information about Titanic passengers, including class, age, fare, and gender.
        - We'll use numeric passenger features to form clusters and explore potential patterns.
        """)
    elif source == "Iris Dataset":
        st.markdown("""
        **Iris Dataset Overview:**
        - This classic dataset contains measurements of iris flowers.
        - We'll cluster the flowers based on their sepal and petal measurements.
        """)
    else:
        st.markdown("""
        **User-Uploaded Dataset:**
        - You uploaded your own dataset! You can now cluster your data using the selected features.
        """)
        
# Tab 2
with tab2:
    st.subheader("Selected Model & Settings")
    st.write(f"**Model:** {model_choice}")
    st.write(f"**Selected Features:** {feature_cols}")
    st.write(f"**X Shape:** {X.shape}")

    if model_choice == "K-Means Clustering":
        st.markdown(f"""
        - **Number of Clusters (k):** {k}  
        - **Initialization Method:** {init_method}

        **K-Means Notes:**
        - K-Means tries to partition data into {k} distinct clusters.
        - The initialization method can impact how quickly the algorithm converges.
        """)
    elif model_choice == "Principal Component Analysis (PCA)":
        st.markdown(f"""
        - **Number of Components:** {n_components}

        **PCA Notes:**
        - PCA reduces dimensionality by finding the main directions of variance in the data.
        - Helps visualize patterns and correlations between features.
        """)