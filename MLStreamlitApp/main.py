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
# Load Sample Data
# -----------------------------------------------
@st.cache_data
def load_sample_data(name):
    if name == 'Iris':
        df = sns.load_dataset('iris')
        X = df.drop('species', axis=1)
        y = df['species']
        return X, y, 'species'
    elif name == 'Titanic':
        df = sns.load_dataset('titanic')
        df.dropna(subset=['age'], inplace=True)
        df = pd.get_dummies(df, columns=['sex'], drop_first=True)
        features = ['pclass', 'age', 'sibsp', 'parch', 'fare', 'sex_male']
        df = df[features + ['survived']]
        X = df[features]
        y = df['survived']
        return X, y, 'survived'
    return None, None, None

# -----------------------------------------------
# Dataset Selection
# -----------------------------------------------
st.sidebar.header("1. Choose Dataset")
dataset_choice = st.sidebar.radio("Select a dataset:", ['Iris', 'Titanic', 'Upload Your Own CSV'])

if dataset_choice == 'Upload Your Own CSV':
    uploaded_file = st.sidebar.file_uploader("Upload CSV", type=['csv'])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.dataframe(df.head())
        target_column = st.sidebar.selectbox("Select target variable", df.columns)
        X = df.drop(columns=[target_column])
        y = df[target_column]
    else:
        st.stop()
else:
    X, y, target_column = load_sample_data(dataset_choice)
    df = pd.concat([X, y], axis=1)
    st.dataframe(df.head())

# Label encode target if needed
original_target = y
if y.dtype == 'object':
    y = LabelEncoder().fit_transform(y)

# -----------------------------------------------
# Model Selection
# -----------------------------------------------
st.sidebar.header("2. Choose Model")
model_name = st.sidebar.selectbox("Model", ['Logistic Regression', 'Decision Tree', 'KNN'])

if model_name == 'Logistic Regression':
    C = st.sidebar.slider("Regularization Strength (C)", 0.01, 10.0, 1.0)
    model = LogisticRegression(C=C)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

elif model_name == 'KNN':
    k = st.sidebar.slider("Number of Neighbors (K)", 1, 15, 5)
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

elif model_name == 'Decision Tree (GridSearchCV)':
    param_grid = {
        'max_depth': list(range(2, 10)),
        'criterion': ['gini', 'entropy']
    }
    base_model = DecisionTreeClassifier(random_state=42)
    grid_search = GridSearchCV(base_model, param_grid, cv=5)
    grid_search.fit(X_train, y_train)
    model = grid_search.best_estimator_
    y_pred = model.predict(X_test)

    st.sidebar.success(f"Best Parameters: {grid_search.best_params_}")

# -----------------------------------------------
# Evaluation
# -----------------------------------------------
st.subheader("📊 Classification Metrics")
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, average='weighted')
rec = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')
st.markdown(f"""
- **Accuracy**: {acc:.2f}  
- **Precision**: {prec:.2f}  
- **Recall**: {rec:.2f}  
- **F1 Score**: {f1:.2f}
""")

# -----------------------------------------------
# Confusion Matrix
# -----------------------------------------------
st.subheader("🔲 Confusion Matrix")
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
st.pyplot(fig)

# -----------------------------------------------
# ROC Curve for Binary Classification
# -----------------------------------------------
if len(np.unique(y)) == 2:
    st.subheader("📈 ROC Curve")
    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    fig2, ax2 = plt.subplots()
    ax2.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    ax2.plot([0, 1], [0, 1], linestyle='--', color='gray')
    ax2.set_xlabel("False Positive Rate")
    ax2.set_ylabel("True Positive Rate")
    ax2.set_title("ROC Curve")
    ax2.legend()
    st.pyplot(fig2)

# -----------------------------------------------
# Visualize Decision Tree (if selected)
# -----------------------------------------------
if "Decision Tree" in model_name:
    st.subheader("🌳 Visualized Decision Tree")
    fig3, ax3 = plt.subplots(figsize=(10, 5))
    plot_tree(model, filled=True, feature_names=X.columns, class_names=True if len(np.unique(y)) > 2 else ['0', '1'], ax=ax3)
    st.pyplot(fig3)