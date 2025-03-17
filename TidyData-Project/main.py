# ================================================================================
# Data Cleaning 
# ================================================================================

import pandas as pd
df = pd.read_csv("TidyData-Project/data/olympics_08_medalists.csv")

# show original data set
print("Original (Untidy) DataFrame:")
print(df)


# melt the dataset into a long format
melted_df = pd.melt(df,
        id_vars=["medalist_name"],
        value_vars=["male_archery", "female_archery", "male_athletics", "female_athletics", "male_badminton", "female_badminton", "male_baseball", "male_basketball",
        "female_basketball", "male_boxing", "male_canoeing and kayaking","female_canoeing and kayaking", "male_road bicycle racing",
        "female_road bicycle racing", "male_track cycling", "female_track cycling","male_mountain biking", "female_mountain biking", "male_bmx", "female_bmx",
        "male_diving", "female_diving", "female_equestrian sport", "male_equestrian sport", "male_fencing", "female_fencing",
        "male_field hockey", "female_field hockey", "male_association football", "female_association football", "male_artistic gymnastics",
        "female_artistic gymnastics", "female_rhythmic gymnastics", "male_trampoline gymnastics", "female_trampoline gymnastics",
        "male_handball", "female_handball", "male_judo", "female_judo", "male_modern pentathlon", "female_modern pentathlon", "male_rowing",
        "female_rowing", "male_sailing", "female_sailing", "male_shooting sport", "female_shooting sport", "female_softball", "male_swimming",
        "female_swimming", "female_synchronized swimming", "male_table tennis",  "female_table tennis", "male_taekwondo", "female_taekwondo",
        "male_tennis", "female_tennis", "male_triathlon", "female_triathlon", "male_beach volleyball", "female_beach volleyball", "male_volleyball",
        "female_volleyball", "male_water polo", "female_water polo", "male_weightlifting", "female_weightlifting", "male_freestyle wrestling",
        "female_freestyle wrestling", "male_greco-roman wrestling"],
        var_name="Gender_Sport",
        value_name="Medal")

# drop rows where there is no medal assigned
melted_df = melted_df.dropna()
print(melted_df)

# use str.split to split gender and sport by the underscore (_), split only the first occurence (n=1),
# and create two new columns for the new variables (expand=True)
melted_df[['Gender', 'Sport']] = melted_df['Gender_Sport'].str.split('_', n=1, expand=True)

# use str.replace to replace underscore (_) with a space in the sport column
melted_df["Sport"] = melted_df["Sport"].str.replace("_", " ", regex=True)

# rename columns
melted_df = melted_df.rename(columns={"medalist_name": "Medalist Name"})
# drop the original Gender_Sport column
melted_df = melted_df.drop(columns=["Gender_Sport"])
# reorder columns
melted_df = melted_df[['Medalist Name', 'Gender', 'Sport', 'Medal']]

# capitalize everything
melted_df["Sport"] = melted_df["Sport"].str.title()
melted_df["Gender"] = melted_df["Gender"].str.title()
melted_df["Medal"] = melted_df["Medal"].str.title()

print("Cleaned (Tidy) DataFrame:")
print(melted_df)

# ================================================================================
# Pivot Table 
# ================================================================================

pivot_table = melted_df.pivot_table(index='Sport', columns='Gender', values='Medal', aggfunc='count', fill_value=0)
print(pivot_table)

# ================================================================================
# Data Visualization
# ================================================================================

# visualization tools
import matplotlib.pyplot as plt
import seaborn as sns

# visualization 1
plt.figure(figsize=(12, 6))
sport_counts = melted_df['Sport'].value_counts().head(5) 
sns.barplot(x=sport_counts.index, y=sport_counts.values, palette = "viridis")
plt.title("Top 5 Sports by Number of Medals")
plt.xlabel("Sport")
plt.ylabel("Number of Medals")
plt.show()
# shows the top 5 sports that won the most medals

# visualization 2 
plt.figure(figsize=(10, 5))
sns.countplot(data=melted_df,
    x="Medal", hue="Gender", order=["Gold", "Silver", "Bronze"],
    palette = {"Male": "dodgerblue", "Female": "hotpink"})
plt.title("Number of Medals by Gender")
plt.xlabel("Medal Type")
plt.ylabel("Number of Medals")
plt.show()
# shows the total number of medals split by gender (male and female) 

# visualization 3
plt.figure(figsize=(6, 6))
plt.pie( melted_df["Medal"].value_counts(), 
    labels=["Gold", "Silver", "Bronze"], 
    autopct='%1.1f%%', 
    colors=["goldenrod", "silver", "sienna"], 
)
plt.title("Medal Distribution")
plt.show()
# shows split of gold, silver, and bronze medals 
