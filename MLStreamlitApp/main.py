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
# Application Information (Updated)
# -----------------------------------------------
st.set_page_config(page_title="Iris Classifier", layout="wide")
st.title("Iris Classifier: KNN & Decision Tree")  # Updated title
st.markdown("""
### Interactive Classification Explorer
Now comparing **K-Nearest Neighbors** and **Decision Tree** algorithms:
- Adjust parameters for each algorithm
- Compare performance metrics
- Visualize decision boundaries
""")

# -----------------------------------------------
# Helper Functions (Added Decision Tree function)
# -----------------------------------------------
@st.cache_data
def load_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['target'] = iris.target
    df['species'] = df['target'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})
    return df, iris.feature_names

def split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def train_knn(X_train, y_train, n_neighbors):
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train, y_train)
    return knn

# NEW: Decision Tree training function
def train_decision_tree(X_train, y_train, max_depth, min_samples_split, criterion):
    dtree = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        criterion=criterion,
        random_state=42
    )
    dtree.fit(X_train, y_train)
    return dtree

def plot_confusion_matrix(cm, classes, title):
    plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    st.pyplot(plt)
    plt.clf()

def plot_feature_relationship(df, x_feat, y_feat):
    plt.figure(figsize=(6,4))
    sns.scatterplot(data=df, x=x_feat, y=y_feat, 
                    hue='species', palette='viridis', s=100)
    plt.title(f"{x_feat} vs {y_feat} by Species")
    st.pyplot(plt)
    plt.clf()

# -----------------------------------------------
# Data Loading and Preparation
# -----------------------------------------------
df, feature_names = load_data()
species_names = df['species'].unique()

# -----------------------------------------------
# Streamlit App Layout 
# -----------------------------------------------
st.sidebar.header("Model Configuration")

# Algorithm selection radio button
algorithm = st.sidebar.radio(
    "Select Algorithm",
    ["K-Nearest Neighbors", "Decision Tree"],
    index=0
)

# Feature selection
selected_features = st.sidebar.multiselect(
    "Select Features", 
    feature_names, 
    default=feature_names[:2]
)

# -----------------------------------------------
# KNN 
# -----------------------------------------------
if algorithm == "K-Nearest Neighbors":
    k = st.sidebar.slider(
        "Number of neighbors (k)", 
        min_value=1, 
        max_value=15, 
        value=5,
        step=2
    )
    scale_data = st.sidebar.checkbox("Scale Features", value=True)
# -----------------------------------------------
# Decision Tree 
# -----------------------------------------------
else:
    max_depth = st.sidebar.slider(
        "Max Tree Depth", 
        min_value=1, 
        max_value=10, 
        value=3
    )
    min_samples_split = st.sidebar.slider(
        "Min Samples Split", 
        min_value=2, 
        max_value=20, 
        value=2
    )
    criterion = st.sidebar.selectbox(
        "Splitting Criterion",
        ["gini", "entropy"]
    )

# Common parameters
test_size = st.sidebar.slider(
    "Test Set Size", 
    min_value=0.1, 
    max_value=0.5, 
    value=0.2, 
    step=0.05
)

# Prepare data
X = df[selected_features]
y = df['target']

if len(selected_features) < 2:
    st.warning("Please select at least 2 features")
    st.stop()

# Split data
X_train, X_test, y_train, y_test = split_data(X, y, test_size=test_size)

# Scale if KNN is selected and scaling is enabled
if algorithm == "K-Nearest Neighbors" and scale_data:
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    data_type = "Scaled"
else:
    data_type = "Unscaled"

# -----------------------------------------------
# Model Training 
# -----------------------------------------------
if st.sidebar.button("Train Model"):
    if algorithm == "K-Nearest Neighbors":
        model = train_knn(X_train, y_train, n_neighbors=k)
        model_name = f"KNN (k={k})"
    else:
        model = train_decision_tree(X_train, y_train, max_depth, min_samples_split, criterion)
        model_name = f"Decision Tree (depth={max_depth})"
    
    y_pred = model.predict(X_test)
    accuracy_val = accuracy_score(y_test, y_pred)

    # -----------------------------------------------
    # Main Display 
    # -----------------------------------------------
    st.header(f"{algorithm} Performance")
    st.write(f"**Configuration:** {data_type} Data | {len(selected_features)} Features")
    
    # Metrics columns
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Accuracy", f"{accuracy_val:.2%}")
    with col2:
        st.metric("Features Used", ", ".join(selected_features))
    
    # Results columns
    col3, col4 = st.columns(2)
    with col3:
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        plot_confusion_matrix(cm, species_names, f"{model_name} Confusion Matrix")
    
    with col4:
        st.subheader("Classification Report")
        report = classification_report(y_test, y_pred, target_names=species_names, output_dict=True)
        st.dataframe(pd.DataFrame(report).transpose())
    
    # Decision Tree visualization
    if algorithm == "Decision Tree":
        st.subheader("Decision Tree Visualization")
        dot_data = tree.export_graphviz(
            model,
            feature_names=selected_features,
            class_names=species_names,
            filled=True,
            rounded=True,
            special_characters=True
        )
        st.graphviz_chart(dot_data)
        
        # Feature importance
        st.subheader("Feature Importance")
        importance = pd.DataFrame({
            'Feature': selected_features,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        st.bar_chart(importance.set_index('Feature'))

# Feature visualization
if len(selected_features) >= 2:
    st.header("Feature Relationships")
    x_feat = st.selectbox("X-axis feature", selected_features, index=0)
    y_feat = st.selectbox("Y-axis feature", selected_features, index=1)
    plot_feature_relationship(df, x_feat, y_feat)

# Data exploration
with st.expander("Dataset Information"):
    st.write("### Iris Dataset Characteristics")
    st.write("""
    The Iris dataset contains measurements for 150 iris flowers from three species:
    - Setosa
    - Versicolor
    - Virginica
    
    Four features were measured for each sample:
    - Sepal length (cm)
    - Sepal width (cm)
    - Petal length (cm)
    - Petal width (cm)
    """)
    
    st.subheader("Dataset Sample")
    st.dataframe(df.head())
    
    st.subheader("Feature Distributions")
    fig, ax = plt.subplots(2, 2, figsize=(10, 8))
    for i, feature in enumerate(feature_names):
        row, col = divmod(i, 2)
        sns.boxplot(data=df, x='species', y=feature, ax=ax[row, col])
    st.pyplot(fig)