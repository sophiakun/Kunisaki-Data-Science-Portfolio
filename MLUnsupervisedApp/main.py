# -----------------------------------------------
# Unsupervised Machine Learning Streamlit App
# -----------------------------------------------

# Launch Streamlit Cloud App:
# requirements.txt file commands: pip install pipreqs

# -----------------------------------------------
# Import Libraries
# -----------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering

# -----------------------------------------------
# Application Information
# -----------------------------------------------

st.title("🔍 Unsupervised Machine Learning Explorer 🔍")
st.markdown("""
Unsupervised machine learning is a type of machine learning that **finds patterns or groupings** in data without using labeled outcomes.  Unlike supervised learning (which trains a model to predict a known target), unsupervised learning, **identifies natural clusters**
within data and can **discovers hidden trends** you might not know exist!
    
Explore different unsupervised machine learning models including:  
1. **K-Means Clustering**: an algorithm that partitions data into distinct clusters  
   by minimizing within-cluster variance
2. **Principal Component Analysis (PCA)**: a dimensionality reduction technique that  
   transforms features into principal components capturing the most variance
3. **Hierarchical (Agglomerative) Clustering**: a bottom-up approach that builds nested clusters  
   by successively merging the closest pairs of clusters
            
Upload **your own dataset** or use the built-in **Iris Dataset** or **Titanic Dataset** to engage with this interactive app!
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

    # Drop rows with missing values
    df = df.dropna()

    # Encode categorical variables
    df = pd.get_dummies(df, drop_first=True)

    # Select numeric columns
    numeric_cols = df.select_dtypes(include="number").columns.tolist()

elif source == "Titanic Dataset":
    df = sns.load_dataset("titanic").dropna(subset=["age"])
    df = pd.get_dummies(df, columns=["sex"], drop_first=True)
    numeric_cols = df.select_dtypes(include="number").columns.tolist()

else:  # Iris Dataset
    df = sns.load_dataset("iris")
    numeric_cols = df.select_dtypes(include="number").columns.tolist()

# -----------------------------------------------
# Feature Selection Sidebar
# -----------------------------------------------

# Create header in sidebar for feature selection
st.sidebar.header("2. Select Features for Clustering")

# Get numeric oclumns from the data
if source == "Iris Dataset":
    prompt = "Select flower measurements to include in clustering:"
    default_features = numeric_cols
elif source == "Titanic Dataset":
    prompt = "Select passenger features to include in clustering:"
    default_features = [col for col in ['pclass', 'age', 'sibsp', 'parch', 'fare', 'sex_male'] if col in numeric_cols]
else:
    prompt = "Select numeric columns to include in clustering:"
    default_features = numeric_cols

if not numeric_cols:
    st.warning("No numeric columns available for clustering.")
    st.stop()

feature_cols = st.sidebar.multiselect(prompt, options=numeric_cols, default=default_features)
X = df[feature_cols]
st.sidebar.write(f"Selected features: {', '.join(X.columns)}")

# Apply feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -----------------------------------------------
# Model Selection Sidebar
# -----------------------------------------------

# Create header in sidebar for model selection
st.sidebar.header("3. Choose a Model")
model_choice = st.sidebar.selectbox("Model:", ["K-Means Clustering", "Hierarchical Clustering", "Principal Component Analysis (PCA)"])

# K-Means settings
if model_choice == "K-Means Clustering":
    k = st.sidebar.slider("Number of clusters (k):", 2, 10, 3)
    init_method = st.sidebar.selectbox("Initialization method:", ["k-means++", "random"])

# PCA settings
if model_choice == "Principal Component Analysis (PCA)":
    max_components = min(len(feature_cols), 10)
    n_components = st.sidebar.slider(
        "Number of components:", min_value=2, max_value=max_components, value=2)

# Hierarchical Clustering settings
if model_choice == "Hierarchical Clustering":
    k = st.sidebar.slider("Number of clusters (k):", 2, 10, 3)
    linkage_method = st.sidebar.selectbox("Linkage method:", ["ward", "complete", "average", "single"])

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
        - This unsupervised ML app will explore clusters based on these numeric features to see if interesting patterns emerge, such as grouping by age/fare/class.
        """)
    # Iris-specific information
    elif source == "Iris Dataset":
        st.markdown("""
        **Iris Dataset Overview:**
        - This classic dataset includes measurements of iris flowers across 3 species:
            - *Setosa*
            - *Versicolor*
            - *Virginica*
        - Each sample has the following numeric measurements:
            - sepal length (cm)
            - sepal width (cm)
            - petal length (cm)
            - petal width (cm)
        - This unsupervised ML app will apply clustering to see how well we can naturally group the flowers based on their measurements—without using species labels.
        """)
    
    else:
        st.markdown("""
        **User-Uploaded Dataset:**
        - Great job uploading a dataset. This app will cluster based on the numeric columns you selected
        - The results will help you explore natural groupings or patterns in your data
        """)

# -------------------------------
# Tab 2: Model Settings
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
        """)

    elif model_choice == "Principal Component Analysis (PCA)":
        # Show PCA settings (number of components)
        st.markdown(f"""
        - **Number of Components:** {n_components}

        ---
        **What is PCA?**
        - Principal Component Analysis (PCA) is a **dimensionality reduction** technique
        - It transforms your data into a new set of axes (principal components) that **capture the most variance** in the data
        - PCA helps **simplify complex data** while retaining important patterns
        - It makes it easier to **visualize high-dimensional data** (for example, plotting the first 2 components)

        **How does PCA work?**
        1️. Identifies directions (components) where data varies the most.
        2️. Projects the data onto those components.
        3️. Orders components so that the **first explains the most variance,** the second explains the next most, etc.

        """)

    elif model_choice == "Hierarchical Clustering":
        st.markdown(f"""
    - **Number of Clusters (k):** {k}  
    - **Linkage Method:** `{linkage_method}`

    ---
    **What is Hierarchical Clustering?**
    - A bottom-up clustering approach where:
        1. Each data point starts as its own cluster.
        2. The two closest clusters are merged.
        3. This repeats until only `k` clusters remain.

    **What does the Linkage Method mean?**
    - `ward`: Minimizes the variance between clusters.
    - `complete`: Considers the farthest points when merging clusters.
    - `average`: Uses the average distance between all pairs of points.
    - `single`: Merges based on the nearest points between clusters.

    This approach is useful when you want to **explore nested or tree-like relationships** between data points.
    """)


# -------------------------------
# Tab 3: Evaluation
# -------------------------------

with tab3:
    st.subheader("Results & Evaluation")

    if model_choice == "K-Means Clustering":
        # Step 1️: Train the K-Means clustering model
        kmeans = KMeans(n_clusters=k, init=init_method, random_state=42)
        labels = kmeans.fit_predict(X)

        # Add cluster labels back into the dataframe for reference
        df['Cluster'] = labels
        # Step 2️: Silhouette Score (Cluster Quality)
        sil_score = silhouette_score(X, labels)

        st.markdown(f"""
        **Silhouette Score:** `{sil_score:.3f}`  
        - Ranges from -1 to 1
        - Closer to 1 means well-defined, separated clusters
        - Around 0 means overlapping clusters
        - Below 0 suggests points may have been assigned to the wrong cluster
        """)

        # Step 3️: 2D Cluster Scatter Plot (PCA projection)
        if X.shape[1] >= 2:
            # Use PCA to reduce the data to 2 components for visualization
            pca = PCA(2)
            components = pca.fit_transform(X)

            st.subheader("Cluster Scatter Plot (PCA 2D Projection)")
            st.markdown("""
            - This plot projects your data onto 2 dimensions using PCA
            - Each color represents a different cluster found by K-Means
            """)

            fig, ax = plt.subplots()
            sns.scatterplot(
                x=components[:, 0],
                y=components[:, 1],
                hue=labels,
                palette="Set1",
                ax=ax
            )
            ax.set_xlabel("Principal Component 1")
            ax.set_ylabel("Principal Component 2")
            ax.set_title("K-Means Clusters (2D Projection)")
            st.pyplot(fig)
            
        # Step 4️: Elbow Plot (to find optimal k)
        st.subheader("Elbow Method Plot")
        st.markdown("""
        - This plot shows **inertia** (within-cluster sum of squares) vs. number of clusters
        - Look for the 'elbow' point where adding more clusters doesn't improve inertia much
        - Helps you choose a good value for `k`
        """)

        distortions = []
        K_range = range(1, 11)  # Test k from 1 to 10

        # Loop over k values to calculate inertia
        for k_val in K_range:
            km = KMeans(n_clusters=k_val, init=init_method, random_state=42)
            km.fit(X)
            distortions.append(km.inertia_)

        fig2, ax2 = plt.subplots()
        ax2.plot(K_range, distortions, marker='o')
        ax2.set_xlabel('Number of Clusters (k)')
        ax2.set_ylabel('Inertia')
        ax2.set_title('Elbow Method for Optimal k')
        st.pyplot(fig2)

    elif model_choice == "Principal Component Analysis (PCA)":
        from sklearn.decomposition import PCA

        # Step 1: Apply PCA to the selected features
        pca = PCA(n_components=n_components)
        components = pca.fit_transform(X)
        explained_var = pca.explained_variance_ratio_

        # Step 2: Display explained variance ratio
        st.write("### Explained Variance Ratio per Component")
        st.markdown("""
        - This shows how much variance each principal component explains
        - Higher variance means that component captures more important patterns in the data
        """)

        for idx, var in enumerate(explained_var):
            st.write(f"Component {idx + 1}: **{var:.4f}**")

        # Step 3️: 2D Scatter Plot of First 2 Components
        if n_components >= 2:
            st.subheader("PCA Scatter Plot (First 2 Components)")
            st.markdown("""
            - This plot shows your data projected onto the first 2 principal components
            - Useful for visualizing patterns, clusters, or trends in your data
            """)

            fig, ax = plt.subplots()
            sns.scatterplot(
                x=components[:, 0],
                y=components[:, 1]
            )
            ax.set_xlabel("Principal Component 1")
            ax.set_ylabel("Principal Component 2")
            ax.set_title("PCA - First 2 Components")
            st.pyplot(fig)

        # Step 4️: Scree Plot (Explained Variance)
        st.subheader("PCA Scree Plot")
        st.markdown("""
        - The scree plot shows the explained variance ratio of each principal component
        - Helps you decide how many components capture most of the data’s variance
        - Look for the point where the curve levels off ('elbow')
        """)

        fig2, ax2 = plt.subplots()
        ax2.plot(range(1, n_components + 1), explained_var, marker='o')
        ax2.set_xlabel("Principal Components")
        ax2.set_ylabel("Explained Variance Ratio")
        ax2.set_title("Scree Plot")
        st.pyplot(fig2)
