# Import necessary libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset from seaborn
df = sns.load_dataset('iris')

# Display the first five rows of the dataset
print("First five rows of the dataset:")
print(df.head())