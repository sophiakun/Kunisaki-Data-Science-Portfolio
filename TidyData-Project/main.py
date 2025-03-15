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