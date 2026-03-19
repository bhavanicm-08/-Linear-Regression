import pandas as pd

data = {
    "Study_Hours": [1, 2, 3, 4, 5, 6, 7],
    "Marks": [30, 40, 50, 60, 70, 80, 90]
}
df = pd.DataFrame(data)
print(df)

from sklearn.model_selection import train_test_split
X = df[["Study_Hours"]]
y = df["Marks"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Predicted:", y_pred)
print("Actual:", y_test.values)
print("Slope (m):", model.coef_)
print("Intercept (b):", model.intercept_)

import matplotlib.pyplot as plt
plt.scatter(X, y)
plt.plot(X, model.predict(X))
plt.xlabel("Study Hours")
plt.ylabel("Marks")
plt.title("Linear Regression Line")
plt.show()