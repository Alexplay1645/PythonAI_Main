import numpy as np
import pandas as pd
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

X, y = make_regression(n_samples=2000, n_features=4, noise=10, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

r2 = model.score(X_test, y_test)

coeff_df = pd.DataFrame({
    "Ознака": [f"x{i+1}" for i in range(X.shape[1])],
    "Коефіцієнт": model.coef_
})

print(coeff_df)
print(f"R²: {r2:.4f}")