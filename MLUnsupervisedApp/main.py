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

# -------------------------------
# Tab 1: About the Dataset
# -------------------------------

# Tab 1 
with tab1:
    # Show a preview of the loaded dataset (first 5 rows)
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # Titantic-specific information
    if source == "Titanic Dataset":
        st.markdown("""
        **Titanic Dataset Overview:**
        - Contains data on passengers aboard the Titanic.
        - Key features include:
            - `pclass`: Passenger class (1st, 2nd, 3rd)
            - `age`: Age of the passenger
            - `fare`: Ticket fare
            - `sex_male`: Gender (converted to numeric: 1 = male, 0 = female)
            - `sibsp`: Number of siblings/spouses aboard
            - `parch`: Number of parents/children aboard
        - We'll explore clusters based on these numeric features to see if interesting patterns emerge, such as grouping by age/fare/class.
        """)
    # Iris-specific information
    elif source == "Iris Dataset":
        st.markdown("""
        **Iris Dataset Overview:**
        - This classic dataset includes measurements of 150 iris flowers across 3 species:
            - *Setosa*
            - *Versicolor*
            - *Virginica*
        - Each sample has the following numeric measurements:
            - `sepal length (cm)`
            - `sepal width (cm)`
            - `petal length (cm)`
            - `petal width (cm)`
        - We'll apply clustering to see how well we can naturally group the flowers based on their measurements—without using species labels.
        """)
    else:
        st.markdown("""
        **User-Uploaded Dataset:**
        - You uploaded your own dataset!
        - We’ll cluster based on the numeric columns you selected.
        - The results will help you explore natural groupings or patterns in your data.
        """)

# -------------------------------
# Tab 2: Model Settings & Info
# -------------------------------

with tab2:
    st.subheader("Selected Model & Settings")

    # Display general model info and selected feature columns
    st.write(f"**Model:** {model_choice}")
    st.write(f"**Selected Features:** {feature_cols}")
    st.write(f"**X Shape:** {X.shape} (Rows: samples, Columns: features)")

    if model_choice == "K-Means Clustering":
        # Show K-Means hyperparameters (k & init method)
        st.markdown(f"""
        - **Number of Clusters (k):** {k}  
        - **Initialization Method:** `{init_method}`

        ---
        **What is K-Means?**
        - K-Means is a clustering algorithm that partitions data into `{k}` distinct groups (clusters).
        - It assigns each data point to the nearest cluster center based on distance (usually Euclidean).

        **What does the Initialization Method mean?**

        - `k-means++`:
            - A **smart initialization** method that spreads out the initial cluster centers.
            - Helps the algorithm converge faster and **reduces the risk of bad clustering.**
        - `random`:
            - Initializes cluster centers completely **at random.**
            - May sometimes result in slower convergence or poor clustering (but still widely used).

        **How does K-Means work?**
        1️. Start with `{k}` initial cluster centers
        2️. Assign each data point to the nearest cluster
        3️. Recalculate cluster centers based on current assignments
        4️. Repeat until the assignments stop changing (convergence)

        **Note:** K-Means is sensitive to the scale of data. That’s why we generally use only numeric features, and scaling helps improve results.
        """)

    elif model_choice == "Principal Component Analysis (PCA)":
        # Show PCA settings (number of components)
        st.markdown(f"""
        - **Number of Components:** {n_components}

        ---
        **What is PCA?**
        - Principal Component Analysis (PCA) is a **dimensionality reduction** technique.
        - It transforms your data into a new set of axes (principal components) that **capture the most variance** in the data.
        
        **Why use PCA?**
        - Helps **simplify complex data** while retaining important patterns.
        - Makes it easier to **visualize high-dimensional data** (for example, plotting the first 2 components).

        **How does PCA work?**
        1️. Identifies directions (components) where data varies the most.
        2️. Projects the data onto those components.
        3️. Orders components so that the **first explains the most variance,** the second explains the next most, etc.

        **Note:** PCA works best when the input features are numeric and scaled.
        """)

# Tab 3