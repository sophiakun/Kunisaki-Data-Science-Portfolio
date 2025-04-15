# -----------------------------------------------
# Launch Streamlit Cloud App
# -----------------------------------------------

# requirements.txt file commands:
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
    confusion_matrix, classification_report, roc_auc_score
)
    
# -----------------------------------------------
# Application Information
# -----------------------------------------------

st.title("⚙️ Interactive Supervised Machine Learning Explorer ⚙️")
st.markdown("""
Explore different supervised machine learning models including:  
1. **Logistic Regression**  
2. **Decision Trees**  
3. **K-Nearest Neighbors (KNN)**  

Upload **your own dataset** or use the built-in **Titanic** and **Iris** datasets to engage with this interactive app!
""")

# -----------------------------------------------
# User-Uploaded Dataset
# -----------------------------------------------

# Create sidebar with different datasets to choose from
st.sidebar.header("1. Choose a Dataset")
source = st.sidebar.radio("Select dataset source:", ["Iris", "Titanic", "Upload your own CSV"])

if source == "Upload your own CSV":
    uploaded_file = st.sidebar.file_uploader("Upload CSV", type="csv")
    if not uploaded_file:
        st.warning("Please upload a CSV file to continue.")
        st.stop()
    df = pd.read_csv(uploaded_file)

else:
    if source == "Titanic":
        df = sns.load_dataset("titanic").dropna(subset=["age"])
        df = pd.get_dummies(df, columns=["sex"], drop_first=True)
        features = ['pclass', 'age', 'sibsp', 'parch', 'fare', 'sex_male']
        X = df[features]
        y = df["survived"]
    else:
        df = sns.load_dataset("iris")
        X = df.drop(columns=["species"])
        y = df["species"]

# -----------------------------------------------
# Feature & Target Selection for User-Uploaded Dataset
# -----------------------------------------------

if source == "Upload your own CSV":
    # Select target column
    target_col = st.sidebar.selectbox("Select the target variable", df.columns)

    # Select features
    feature_cols = st.sidebar.multiselect(
        "Select features to include in the model",
        options=[col for col in df.columns if col != target_col],
        default=[col for col in df.columns if col != target_col]
    )

    if not feature_cols:
        st.error("Please select at least one feature column.")
        st.stop()

    # Only keep numeric features
    X = df[feature_cols].select_dtypes(include="number")
    y = df[target_col]

    if X.empty:
        st.error("Selected features must be numeric.")
        st.stop()

    st.write(f"Selected features: {', '.join(X.columns)}")

# -----------------------------------------------
# Model Selection Sidebar
# -----------------------------------------------

st.sidebar.header("2. Choose a Model")
model_name = st.sidebar.selectbox("Model:", ["Logistic Regression", "Decision Tree", "KNN"])

if model_name == "Logistic Regression":
    model = LogisticRegression()
elif model_name == "Decision Tree":
    model = DecisionTreeClassifier()
else:
    model = KNeighborsClassifier()

# ----------------------------
# Data Preprocessing and Splitting
# ----------------------------

# Make 3 tabs 
tab1, tab2, tab3 = st.tabs(["Dataset Preview", "Model Settings", "Results & Evaluation"])

with tab1:
    st.subheader("Preview of Data")
    st.dataframe(df.head())

with tab2:
    st.subheader("Model Info and Configuration")
    st.write("Selected Model:", model_name)
    st.write("X Shape:", X.shape)
    st.write("y Distribution:", pd.Series(y).value_counts())

with tab3:
    st.subheader("Training and Evaluation")

    # Encode categorical features in X and y
    X = pd.get_dummies(X, drop_first=True)
    if y.dtype == "object":
        y = LabelEncoder().fit_transform(y)

    # Split test and training subsets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale dataset for logistic and k-nearest neighbors models
    if model_name in ["Logistic Regression", "KNN"]:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    # Fit model and make prediction
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

# ----------------------------
# Metrics
# ----------------------------

    # Display ROC AUC for binary classifiers (excluding Decision Tree)
    st.subheader("Model Performance Metrics")
    if len(np.unique(y)) == 2 and model_name != "Decision Tree":
        st.markdown(f"- **ROC AUC Score:** {roc_auc_score(y_test, y_pred):.2f}")

    st.markdown(f"""
    - **Accuracy:** {accuracy_score(y_test, y_pred):.2f}  
    - **Precision:** {precision_score(y_test, y_pred, average='weighted', zero_division=0):.2f}  
    - **Recall:** {recall_score(y_test, y_pred, average='weighted', zero_division=0):.2f}  
    - **F1 Score:** {f1_score(y_test, y_pred, average='weighted', zero_division=0):.2f}
    """)

    # Confusion Matrix
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

    # Classification Report Table
    st.subheader("Classification Report")
    report_df = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).transpose()
    st.dataframe(report_df.style.format("{:.2f}"))
    
    # Show logistic regression model coefficients
    if model_name == "Logistic Regression":
        st.subheader("Model Coefficients")
    
    # Make sure feature names align with scaled or encoded input
        if isinstance(X_train, np.ndarray):
         feature_names = X.columns
        else:
            feature_names = X_train.columns

        coef_df = pd.DataFrame({
            "Feature": feature_names,
            "Coefficient": model.coef_[0]
        })

        coef_df["Abs(Coefficient)"] = coef_df["Coefficient"].abs()
        coef_df = coef_df.sort_values("Abs(Coefficient)", ascending=False)

        st.dataframe(coef_df[["Feature", "Coefficient"]].style.format("{:.4f}"))

    # Iris Dataset Visualization
    if source == "Iris":
        st.subheader("Pairplot of Iris Features")
        fig = sns.pairplot(df, hue="species", palette="viridis")
        st.pyplot(fig)
