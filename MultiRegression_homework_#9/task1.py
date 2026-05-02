import pandas as pd
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

X, y = make_regression(n_samples=2000, n_features=4, noise=10, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)

coefficients = model.coef_

df_coeff = pd.DataFrame({
    "Ознака": ["x1", "x2", "x3", "x4"],
    "Коефіцієнт": coefficients
})

print("Коефіцієнти моделі:")
print(df_coeff)

print("\nR² на тестовій вибірці:", r2)