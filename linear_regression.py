# LINEAR REGRESSION PRACTICAL

# STEP 1: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder


# STEP 2: Load Dataset
df = pd.read_csv("smart_class_energy_data.csv")

print("First 5 rows:")
print(df.head())

print("\nDataset Information:")
print(df.describe())


# STEP 3: Remove Missing Values
df = df.dropna()

print("\nAfter Removing Missing Values:")
print(df.isnull().sum())


# STEP 4: Convert Time_of_Day into Numbers
le = LabelEncoder()
df["Time_of_Day"] = le.fit_transform(df["Time_of_Day"])

print("\nAfter Label Encoding:")
print(df.head())


# STEP 5: Find Correlation
correlation_matrix = df.corr()

print("\nCorrelation Matrix:")
print(correlation_matrix)


# STEP 6: Display Heatmap
plt.figure(figsize=(10, 6))

sns.heatmap(
    correlation_matrix,
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    linewidths=0.5
)

plt.title("Correlation Matrix Heatmap")
plt.show()


# STEP 7: Remove Outliers using IQR
def remove_outliers_iqr(df):
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)

    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    return df[
        ~((df < lower_bound) | (df > upper_bound)).any(axis=1)
    ]


df_cleaned = remove_outliers_iqr(df)

print("\nOriginal Data Shape:", df.shape)
print("Cleaned Data Shape:", df_cleaned.shape)


# STEP 8: Correlation After Outlier Removal
correlation_matrix = df_cleaned.corr()

plt.figure(figsize=(10, 6))

sns.heatmap(
    correlation_matrix,
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    linewidths=0.5
)

plt.title("Correlation Matrix After Outlier Removal")
plt.show()


# STEP 9: Compare Original and Cleaned Data
plt.figure(figsize=(10, 5))

plt.plot(
    df.index,
    df.mean(axis=1),
    label="Original Data",
    linestyle="dashed"
)

plt.plot(
    df_cleaned.index,
    df_cleaned.mean(axis=1),
    label="Without Outliers"
)

plt.xlabel("Index")
plt.ylabel("Mean Value of Rows")
plt.title("Before and After Removing Outliers")

plt.legend()
plt.grid(True)
plt.show()


# STEP 10: Separate Input and Output
x = df.drop(columns=["AC_Temp"])
y = df["AC_Temp"]

print("\nInput Data:")
print(x.head())

print("\nOutput Data:")
print(y.head())


# STEP 11: Split Dataset into Training and Testing
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=20, random_state=40
)

print("\nTraining Data:", x_train.shape)
print("Testing Data:", x_test.shape)


# STEP 12: Create Linear Regression Model
model = LinearRegression()


# STEP 13: Train the Model
model.fit(x_train, y_train)

print("\nModel Training Completed!")


# STEP 14: Predict Values
y_pred = model.predict(x_test)

print("\nPredicted Values:")
print(y_pred)


# STEP 15: Calculate Mean Squared Error
mse = mean_squared_error(y_test, y_pred)

print("\nMean Squared Error:", mse)


# STEP 16: Calculate R2 Score
r2 = r2_score(y_test, y_pred)

print("R2 Score:", r2)


# STEP 17: Predict for New Sample
new_sample = pd.DataFrame(
    [[55, 32.145236, 2, 85.456921, 0, 1, 3]],
    columns=x.columns
)

prediction = model.predict(new_sample)

print("\nPredicted AC Temperature:", prediction[0])