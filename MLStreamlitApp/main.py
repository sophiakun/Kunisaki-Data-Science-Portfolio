# -----------------------------------------------
# Launch Streamlit Cloud App
# -----------------------------------------------

# requirements.txt file commands: pip install pipreqs

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

st.title(" ⚙️ Interactive Supervised Machine Learning Explorer ⚙️")
st.markdown("""
Explore different supervised machine learning models including:  
1. **Logistic Regression**: a statistical model used for binary classification,
            estimating the probability of target variable gieven a set of features
2. **Decision Trees**:  a flowchart-like model that splits data into branches based on
            features and can handle both numerical and categorical data
3. **K-Nearest Neighbors (KNN)**: a non-parametric model that classifies a data point
            based on the majority class among its ‘k’ nearest neighbors in the training set

Upload **your own dataset** or use the built-in **Titanic** and **Iris** datasets to engage with this interactive app!
""")

# -----------------------------------------------
# User-Uploaded Dataset
# -----------------------------------------------

# Create sidebar with different datasets to choose from
st.sidebar.header("1. Choose a Dataset")
source = st.sidebar.radio("Select dataset source:", ["Iris", "Titanic", "Upload your own CSV"])
# for user-uploaded CSV
if source == "Upload your own CSV":
    uploaded_file = st.sidebar.file_uploader("Upload CSV", type="csv")
    if not uploaded_file:
        st.warning("Please upload a CSV file to continue.")
        st.stop()
    df = pd.read_csv(uploaded_file)

# for the Titanic example
else:
    if source == "Titanic":
        df = sns.load_dataset("titanic").dropna(subset=["age"])
        df = pd.get_dummies(df, columns=["sex"], drop_first=True)
        features = ['pclass', 'age', 'sibsp', 'parch', 'fare', 'sex_male']
        X = df[features]
        y = df["survived"]
    else: # for the Iris example
        df = sns.load_dataset("iris")
        X = df.drop(columns=["species"])
        y = df["species"]

# -----------------------------------------------
# Feature & Target Selection for User-Uploaded Dataset
# -----------------------------------------------

if source == "Upload your own CSV":
    # Select target column
    target_col = st.sidebar.selectbox("Select the target variable (what you want to predict)", df.columns)

    # Select features
    feature_cols = st.sidebar.multiselect(
        "Select features to include in the model (inputs used to make predictions)",
        options=[col for col in df.columns if col != target_col], # Dropdown options: all columns except the target
        default=[col for col in df.columns if col != target_col] # Default options: pre-select all available features (excluding the target)
    )

    # Keep only numeric columns from the selected feature columns
    X = df[feature_cols].select_dtypes(include="number")
    # Extract the target column (labels) as the response variable
    y = df[target_col]
    
    # Display the list of selected feature names to the user
    st.write(f"Selected features: {', '.join(X.columns)}")

# -----------------------------------------------
# Model Selection Sidebar
# -----------------------------------------------

# Allow user to choose the type of model they want
st.sidebar.header("2. Choose a Model")
model_name = st.sidebar.selectbox("Model:", ["Logistic Regression", "Decision Tree", "KNN"])

# If Decision Tree selected, user can control tree depth (avoid overfitting)
# and require a minimum number of samples for splits (control tree complexity)
if model_name == "Decision Tree":
    max_depth = st.sidebar.slider("Choose max depth of tree:", min_value=1, max_value=20, value=5)
    min_samples_split = st.sidebar.slider("Choose minimum samples to split:", min_value=2, max_value=10, value=2)

# If KNN selected, show a slider for 'k'
if model_name == "KNN":
    selected_k = st.sidebar.slider(
        "Choose number of neighbors (k):",
        min_value=1, max_value=15, step=2, value=5
    )

# Initialize respective models
if model_name == "Logistic Regression":
    model = LogisticRegression()
elif model_name == "Decision Tree":
    model = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=42)
else:  # KNN
    model = KNeighborsClassifier(n_neighbors=selected_k)

# ----------------------------
# Data Preprocessing and Splitting
# ----------------------------

# Make 3 tabs 
tab1, tab2, tab3 = st.tabs(["About", "Model Settings", "Evaluation"])

with tab1:
    # Show "About the Dataset" only for built-in datasets
    if source == "Titanic":
        st.markdown("""
        ### Titanic Dataset
        - **Goal:** Predict whether a passenger survived the Titanic
        - **Target:** `survived` (1 = survived, 0 = did not survive)
        - **Features:** Passenger class, age, family aboard, fare, and gender
        - **Evaluation Metrics:**
            - **Accuracy:** % of passengers correctly predicted
            - **Precision:** Of predicted survivors, how many actually survived?
            - **Recall:** Of actual survivors, how many did the model correctly predict as survivors?
            - **F1 Score:** Balance between precision and recall
            - **Confusion Matrix:** Shows true/false predictions per class
        """)
        st.subheader("Dataset Preview")
        st.dataframe(df[features + ["survived"]].head())

    elif source == "Iris":
        st.markdown("""
        ### Iris Dataset
        - **Goal:** Predict the species of an iris flower based on petal and sepal measurements
        - **Target:** `species` (Setosa, Versicolor, Virginica)
        - **Features:** Sepal and petal length and width
        - **Evaluation Metrics:**
            - **Accuracy:** % of flowers correctly classified
            - **Precision/Recall/F1:** Show model quality per class
            - **Confusion Matrix:** Reveals which species are misclassified
            - **Pairplot:** Visualizes how features cluster by species
        """)
        st.subheader("Dataset Preview")
        st.dataframe(df.head())

        st.subheader("Iris Dataset Pairplot")
        st.markdown("""
        This plot helps visualize how different flower species separate based on their features:

        - Each dot is a flower  
        - Colors represent the species  
        - You can see how well the features help separate classes
        """)
        fig = sns.pairplot(df, hue="species", palette="viridis")
        st.pyplot(fig)

    else:
        # User-uploaded dataset – no "About the Dataset" header
        st.subheader("Dataset Preview")
        st.dataframe(df.head())

with tab2:
    st.markdown("""
    - **X Shape:** Rows = number of samples; Columns = number of input features
    - **y Distribution:** Shows how many of each type of target variable are present
    """)
    st.subheader("Model Info and Configuration")
    st.write("Selected Model:", model_name)
    st.write("X Shape:", X.shape)
    st.write("y Distribution:", pd.Series(y).value_counts())

with tab3:
    # Encode categorical features and labels
    X = pd.get_dummies(X, drop_first=True)
    if y.dtype == "object":
        y = LabelEncoder().fit_transform(y)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale for Logistic Regression and KNN
    if model_name in ["Logistic Regression", "KNN"]:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    # Train and predict
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Show user-selected hyperparameters
    if model_name == "KNN":
        st.markdown(f""" **KNN Configuration**
    - **k (Number of Neighbors):** `{selected_k}`  
      The number of closest data points the algorithm uses to classify a new point.  
      A smaller `k` may capture noise (overfitting), while a larger `k` makes the model more generalized.""")

    elif model_name == "Decision Tree":
        st.markdown(f"""
**Decision Tree Configuration**

- **Max Depth:** `{max_depth}`  
  This limits how deep the tree can go. Lower values create simpler trees; higher values allow more splits but can lead to overfitting.

- **Min Samples to Split:** `{min_samples_split}`  
  The minimum number of samples required to split a node. Larger values restrict growth and help prevent overfitting.
""")

# ----------------------------
# Metrics
# ----------------------------

    # Display ROC AUC for binary classifiers (excluding Decision Tree)
    st.subheader("Model Performance Metrics")
    st.markdown("""
    - **Accuracy:** % of total correct predictions  
    - **Precision:** % of positive predictions that were correct  
    - **Recall:** % of actual positives that were correctly predicted  
    - **F1 Score:** Balance between precision and recall  
    - **ROC AUC:** (for binary models) how well the model distinguishes classes
    """)

    # Compute ROC AUC if the problem is binary classification and model is not a Decision Tree
    if len(np.unique(y)) == 2 and model_name != "Decision Tree":
        st.markdown(f"- **ROC AUC Score:** {roc_auc_score(y_test, y_pred):.2f}") # Show ROC AUC score (2 decimal places)

    st.markdown(f"""
    - **Accuracy:** {accuracy_score(y_test, y_pred):.2f}  
    - **Precision:** {precision_score(y_test, y_pred, average='weighted', zero_division=0):.2f}  
    - **Recall:** {recall_score(y_test, y_pred, average='weighted', zero_division=0):.2f}  
    - **F1 Score:** {f1_score(y_test, y_pred, average='weighted', zero_division=0):.2f}
    """)

    if source == "Iris":
        st.markdown("""
    All metrics from the **Iris Dataset** equal 1.00 (100%)

    The Iris dataset is very well-behaved for machine learning:

    - The classes (Setosa, Versicolor, Virginica) are clearly separated
    - There is no missing or noisy data

    **Perfect classification** is achieved on the test split.
    """)

    # Confusion Matrix
    st.subheader("Confusion Matrix")
    st.markdown("""
This shows how many values were correctly and incorrectly classified:

- **Actual:** The real label from the data  
- **Predicted:** What the ML model guessed  

In the **Titanic example**:  
- `0` = Did **not** survive  
- `1` = **Survived**

The confusion matrix cells represent:
- **True Negatives** (Top-left): Actual = 0, Predicted = 0  
- **False Positives** (Top-right): Actual = 0, Predicted = 1  
- **False Negatives** (Bottom-left): Actual = 1, Predicted = 0  
- **True Positives** (Bottom-right): Actual = 1, Predicted = 1
""")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots() # Create a Matplotlib heatmap
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

    # Classification Report Table
    st.subheader("Classification Report")
    st.markdown("Detailed breakdown of precision, recall, and F1 per class")

    # Generate a classification report and convert it into a formatted DF
    report_df = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).transpose()
    st.dataframe(report_df.style.format("{:.2f}")) # Display as styled table
    
    # Show logistic regression model coefficients
    if model_name == "Logistic Regression":
        st.subheader("Model Coefficients")
        st.markdown("""
These values show how much each feature (input variable) influences the model’s predictions:

- A **positive coefficient** means the feature increases the chance of predicting class `1`  
- A **negative coefficient** means it decreases that chance  
- The **larger the number**, the stronger the influence
""")
        # Extract feature names and corresponding model coefficients
        feature_names = X.columns
        coefficients = model.coef_[0]
        # Create a DataFrame to display features and their corresponding coefficients
        coef_df = pd.DataFrame({
            "Feature": feature_names,
            "Coefficient": pd.Series(coefficients, dtype="float")
        })

        # Drop NaNs 
        coef_df = coef_df.dropna(subset=["Coefficient"])
        # Take absolute value of coefficients and sort by magnitude
        coef_df["Abs(Coefficient)"] = coef_df["Coefficient"].abs()
        coef_df = coef_df.sort_values("Abs(Coefficient)", ascending=False)

        st.dataframe(coef_df[["Feature", "Coefficient"]].style.format({"Coefficient": "{:.4f}"})) # show in app
