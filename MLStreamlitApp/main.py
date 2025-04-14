# -----------------------------------------------
# Launch Streamlit Cloud App
# -----------------------------------------------

# # requirements.txt file commands:
# pip install pipreqs

# import libraries 
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc
)
from sklearn.decomposition import PCA
    
# -----------------------------------------------
# Application Information
# -----------------------------------------------

st.title("Interactive Supervised Machine Learning Explorer")
st.markdown("""
Explore different supervised machine learning models including:  
1. **Logistic Regression**  
2. **Decision Trees**  
3. **K-Nearest Neighbors (KNN)**  

Upload **your own dataset** or use the built-in **Titanic** and **Iris** datasets to engage with this interactive app!
""")

# -----------------------------------------------
# Select Dataset
# -----------------------------------------------
st.sidebar.header("1. Choose a Dataset")
dataset_option = st.sidebar.radio("Select dataset source:", ["Iris", "Titanic", "Upload your own CSV"])

if dataset_option == "Upload your own CSV":
    uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.subheader("Uploaded Dataset Preview")
        st.dataframe(df.head())
        target_column = st.sidebar.selectbox("Select your target variable", df.columns)
        X = df.drop(columns=[target_column])
        y = df[target_column]
    else:
        st.warning("Please upload a CSV file.")
        st.stop()

elif dataset_option == "Titanic":
    df = sns.load_dataset("titanic").dropna(subset=["age"])
    df = pd.get_dummies(df, columns=["sex"], drop_first=True)
    features = ['pclass', 'age', 'sibsp', 'parch', 'fare', 'sex_male']
    X = df[features]
    y = df["survived"]
    st.subheader("Titanic Dataset Preview")
    st.dataframe(df[features + ["survived"]].head())

else:  # Iris
    df = sns.load_dataset("iris")
    X = df.drop(columns=["species"])
    y = df["species"]
    st.subheader("Iris Dataset Preview")
    st.dataframe(df.head())

# ----------------------------
# Model selection
# ----------------------------
st.sidebar.header("2. Choose a Model")
model_choice = st.sidebar.selectbox("Model:", ["Logistic Regression", "Decision Tree", "KNN"])

if model_choice == "Logistic Regression":
    model = LogisticRegression()
elif model_choice == "Decision Tree":
    model = DecisionTreeClassifier()
else:
    model = KNeighborsClassifier()

# ----------------------------
# Data Preprocessing and Splitting
# ----------------------------

# Encode categorical features in X and y
X = pd.get_dummies(X, drop_first=True)

if y.dtype == 'object':
    y = LabelEncoder().fit_transform(y)

# Split dataset into training and testing subsets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale dataset for logistic and k-nearest neighbors models
if model_choice in ["Logistic Regression", "KNN"]:
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

# Make prediction
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# ----------------------------
# Display results
# ----------------------------

# For binary classification show ROC and AUC
if len(np.unique(y)) == 2 and model_choice != "Decision Tree":
    st.write("ROC AUC Score:", roc_auc_score(y_test, y_pred))

# Only show metrics if they are not equal to 1
if not all(score == 1.0 for score in [accuracy_score(y_test, y_pred), 
                                    precision_score(y_test, y_pred, average='weighted'),
                                    recall_score(y_test, y_pred, average='weighted'),
                                    f1_score(y_test, y_pred, average='weighted')]):
    st.markdown(f"""
    - **Accuracy:** {accuracy_score(y_test, y_pred):.2f}  
    - **Precision:** {precision_score(y_test, y_pred, average='weighted'):.2f}  
    - **Recall:** {recall_score(y_test, y_pred, average='weighted'):.2f}  
    - **F1 Score:** {f1_score(y_test, y_pred, average='weighted'):.2f}
    """)
else:
    st.warning("All metrics are 1.0 - the model predicts perfectly on this test set")

st.subheader("Confusion Matrix")
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
st.pyplot(fig)

# Print the classification report with detailed metrics
st.text("\nClassification Report:\n" + classification_report(y_test, y_pred))

# ----------------------------
# Visual for Iris Data
# ----------------------------
if dataset_option == "Iris":
    st.subheader("Visualization")
    
    # Create a pairplot colored by species
    fig = sns.pairplot(df, hue="species", palette="viridis")
    st.pyplot(fig)
