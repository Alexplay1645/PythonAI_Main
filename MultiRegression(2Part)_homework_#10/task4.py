import numpy as np
import pandas as pd
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

X_cv, y_cv = make_regression(n_samples=2000, n_features=4, noise=15, random_state=42)

scaler = StandardScaler()
X_cv_scaled = scaler.fit_transform(X_cv)

model = LinearRegression()

scores = cross_val_score(model, X_cv_scaled, y_cv, cv=5, scoring="r2")

print(f"Середній R²: {scores.mean():.4f}")
print(f"Стандартне відхилення: {scores.std():.4f}")