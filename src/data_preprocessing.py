# ================================
# DATA PREPROCESSING FILE
# ================================

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# --------- Load Dataset ----------
df = pd.read_csv("data/housing.csv")

print("Original Shape:", df.shape)

# --------- Handle Missing Values ----------
df['total_bedrooms'].fillna(df['total_bedrooms'].median(), inplace=True)

# --------- Label Encoding ----------
le = LabelEncoder()
df['ocean_proximity'] = le.fit_transform(df['ocean_proximity'])

# --------- Outlier Removal (IQR Method) ----------
Q1 = df['median_house_value'].quantile(0.25)
Q3 = df['median_house_value'].quantile(0.75)
IQR = Q3 - Q1

df = df[
    (df['median_house_value'] >= Q1 - 1.5 * IQR) &
    (df['median_house_value'] <= Q3 + 1.5 * IQR)
]

print("Shape After Outlier Removal:", df.shape)

# --------- Save Cleaned Data ----------
df.to_csv("data/cleaned_housing.csv", index=False)

print("✅ Cleaned data saved as cleaned_housing.csv")
