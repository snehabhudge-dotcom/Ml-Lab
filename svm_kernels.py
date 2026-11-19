# SVM KERNELS PRACTICAL

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# 1. Generate Dataset

x = np.linspace(-5.0, 5.0, 100)
y = np.sqrt(10**2 - x**2)

y = np.hstack([y, -y])
x = np.hstack([x, -x])

x1 = np.linspace(-5.0, 5.0, 100)
y1 = np.sqrt(5**2 - x1**2)

y1 = np.hstack([y1, -y1])
x1 = np.hstack([x1, -x1])

# 2. Visualize Dataset

plt.scatter(y, x, label="Class 0")
plt.scatter(y1, x1, label="Class 1")

plt.xlabel("X1")
plt.ylabel("X2")
plt.title("SVM Dataset")
plt.legend()
plt.show()

# 3. Create DataFrame

df1 = pd.DataFrame(
    np.vstack([y, x]).T,
    columns=['X1', 'X2']
)

df1['Y'] = 0

df2 = pd.DataFrame(
    np.vstack([y1, x1]).T,
    columns=['X1', 'X2']
)

df2['Y'] = 1

df = pd.concat([df1, df2], ignore_index=True)

print(df.head())

# 4. Independent and Dependent Features

X = df.iloc[:, :2]
target = df['Y']

# 5. Split Dataset into Train and Test

X_train, X_test, y_train, y_test = train_test_split(
    X, target,
    test_size=0.25,
    random_state=0
)

# 6. Linear Kernel

classifier = SVC(kernel="linear")

classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

print("\nLinear Kernel Accuracy:")
print(accuracy_score(y_test, y_pred))

# 7. Polynomial Features

df['X1_Square'] = df['X1']**2
df['X2_Square'] = df['X2']**2
df['X1*X2'] = df['X1'] * df['X2']

X = df[['X1', 'X2', 'X1_Square',
        'X2_Square', 'X1*X2']]

target = df['Y']

# 8. Split Dataset Again

X_train, X_test, y_train, y_test = train_test_split(
    X, target,
    test_size=0.25,
    random_state=0
)

# 9. 3D Visualization

fig = px.scatter_3d(
    df,
    x='X1_Square',
    y='X2_Square',
    z='X1*X2',
    color='Y'
)

fig.show()

# 10. RBF Kernel

classifier = SVC(kernel="rbf")

classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

print("\nRBF Kernel Accuracy:")
print(accuracy_score(y_test, y_pred))