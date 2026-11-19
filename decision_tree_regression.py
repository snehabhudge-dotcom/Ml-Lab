# Decision Tree Regression and 5-Fold Cross Validation
# Install first: python -m pip install pandas numpy matplotlib scikit-learn

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 1. Read the CSV file (keep the CSV in the same folder as this Python file)
df = pd.read_csv("decision_tree_regression_practice.csv")

print("First five rows:")
print(df.head())
print("\nDataset shape:", df.shape)
print("\nMissing values:\n", df.isnull().sum())

# 2. Separate independent variables (X) and dependent variable (y)
X = df.drop("MEDV", axis=1)  # input features
y = df["MEDV"]               # value to predict

# 3. Split data: 67% training and 33% testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42
)

# 4. Create and train a basic Decision Tree Regressor
model = DecisionTreeRegressor(random_state=42)
model.fit(X_train, y_train)

# 5. Predict test-set values and evaluate
y_pred = model.predict(X_test)
print("\nBasic Decision Tree Results")
print("R2 score:", r2_score(y_test, y_pred))
print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))

# 6. Hyperparameter tuning using GridSearchCV with 5-fold cross-validation
parameters = {
    "criterion": ["squared_error", "friedman_mse", "absolute_error"],
    "splitter": ["best", "random"],
    "max_depth": [2, 3, 4, 5, 6, 8, None],
    "max_features": [None, "sqrt", "log2"]
}

grid = GridSearchCV(
    estimator=DecisionTreeRegressor(random_state=42),
    param_grid=parameters,
    cv=5,
    scoring="neg_mean_squared_error",
    n_jobs=-1
)
grid.fit(X_train, y_train)

print("\nBest parameters:", grid.best_params_)
print("Best CV negative MSE:", grid.best_score_)

# 7. Evaluate the best model on the untouched test set
best_model = grid.best_estimator_
best_pred = best_model.predict(X_test)

print("\nTuned Decision Tree Results")
print("R2 score:", r2_score(y_test, best_pred))
print("Mean Absolute Error:", mean_absolute_error(y_test, best_pred))
print("Mean Squared Error:", mean_squared_error(y_test, best_pred))

# 8. Optional: show 5-fold R2 scores for the selected model on training data
cv_scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring="r2")
print("\n5-fold R2 scores:", cv_scores)
print("Average 5-fold R2:", cv_scores.mean())

# 9. Optional: plot the first few levels of the decision tree
plt.figure(figsize=(18, 8))
plot_tree(
    best_model,
    feature_names=X.columns,
    filled=True,
    max_depth=2,
    rounded=True
)
plt.title("Decision Tree Regressor (first levels)")
plt.tight_layout()
plt.show()
