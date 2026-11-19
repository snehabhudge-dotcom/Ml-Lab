# Step 1: Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

# Step 2: Load the dataset
df_black_friday_sales = pd.read_csv("BlackFriday.csv")

# Display first five rows
print(df_black_friday_sales.head())
# Check whether columns contain missing values
print(df_black_friday_sales.isna().any())

# Count total missing values in each column
print(df_black_friday_sales.isnull().sum())
# Fill missing values in Product_Category_2
df_black_friday_sales["Product_Category_2"] = (
    df_black_friday_sales["Product_Category_2"]
    .fillna(-2.0)
    .astype("float32")
)

# Fill missing values in Product_Category_3
df_black_friday_sales["Product_Category_3"] = (
    df_black_friday_sales["Product_Category_3"]
    .fillna(-2.0)
    .astype("float32")
)

# Fill any remaining missing numeric values with column means
df_black_friday_sales.fillna(
    df_black_friday_sales.mean(numeric_only=True),
    inplace=True
)

print(df_black_friday_sales.head())
# Count duplicate rows
print(
    "Duplicate rows:",
    df_black_friday_sales.duplicated().sum()
)

# Remove duplicate rows
df = df_black_friday_sales
df_cleaned = df.drop_duplicates()

# Reset the index
df_cleaned = df_cleaned.reset_index(drop=True)

print("Shape after removing duplicates:", df_cleaned.shape)
# Create scaler
scaler = MinMaxScaler()

# Normalize Purchase values between 0 and 1
df_black_friday_sales["Purchase"] = scaler.fit_transform(
    df_black_friday_sales[["Purchase"]]
)

print(df_black_friday_sales.head())
# Map age groups to meaningful names
age_mapping = {
    "0-17": "toddler",
    "18-25": "young_adult",
    "26-35": "adult",
    "36-45": "middle_age",
    "46-50": "mid_age",
    "51-55": "older_adult",
    "55+": "elderly"
}

# Create a new age column
df_black_friday_sales["Age_new"] = (
    df_black_friday_sales["Age"].map(age_mapping)
)

# Convert age categories into dummy variables
df_black_friday_sales = pd.get_dummies(
    df_black_friday_sales,
    columns=["Age_new"],
    drop_first=True
)

# Encode city and stay duration
cols = ["City_Category", "Stay_In_Current_City_Years"]

for col in cols:
    le = LabelEncoder()
    df_black_friday_sales[col] = le.fit_transform(
        df_black_friday_sales[col]
    )

print(df_black_friday_sales.head())
# Display dataset dimensions
print("Dimensions of dataset:", df_black_friday_sales.shape)

# Display dataset information
df_black_friday_sales.info()

# Find unique values
print("Unique Gender values:",
      df_black_friday_sales["Gender"].unique())

print("Unique Age values:",
      df_black_friday_sales["Age"].unique())

print("Unique City Categories:",
      df_black_friday_sales["City_Category"].unique())
# M = 0 and F = 1
df_black_friday_sales["Gender"] = (
    df_black_friday_sales["Gender"].replace({
        "M": 0,
        "F": 1
    })
)

print(df_black_friday_sales["Gender"].head())
# Statistical summary
print(df_black_friday_sales.describe())

# Display data types
print("Data Types:")
print(df_black_friday_sales.dtypes)

# Unique city category values
print("Unique City Categories:")
print(df_black_friday_sales["City_Category"].unique())
# Plot Gender versus Purchase
sns.barplot(
    data=df_black_friday_sales,
    x="Gender",
    y="Purchase"
)

plt.title("Gender VS Purchase Distribution Plot")
plt.xlabel("Gender (0 = Male, 1 = Female)")
plt.ylabel("Normalized Purchase")
plt.show()