# ================================
# Import libraries
# ================================

import streamlit as st
import pandas as pd

# ================================
# Navigate in terminal
# ================================

# ls
# cd basic-streamlit-app/
# streamlit run main.py

# ================================
# Create title & description
# ================================

st.title("Palmer's Penguins")
st.markdown("The Palmer's Penguins dataset shows species, island, bill length, bill depth, flipper length, body mass, sex, and year for over 300 penguins.")
st.markdown("## Explore Penguin Data by Species")

# ================================
# Load sample dataframe
# ================================

df = pd.read_csv("data/penguins.csv")
st.write("Here's the dataset:")
st.dataframe(df)

# ================================
# Filtering options
# ================================

species = st.selectbox("Select a species", df["species"].unique())
filtered_df = df[(df["species"] == species)]
st.write(f"You selected the {species} species!")
st.dataframe(filtered_df)

# ================================
# Slider using https://cheat-sheet.streamlit.app/
# ================================

mass = st.slider('Slide me', min_value=float(df["body_mass_g"].min()), max_value=float(df["body_mass_g"].max()))
st.write(f"You selected a body mass of {mass} grams")

# ================================
# End of code
# See updated repository README
# ================================

