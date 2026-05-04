import numpy as np
import pandas as pd
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

model1 = LinearRegression()
model1.fit(X_train, y_train)
r2_no_scaling = model1.score(X_test, y_test)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model2 = LinearRegression()
model2.fit(X_train_scaled, y_train)
r2_scaled = model2.score(X_test_scaled, y_test)

scaling_results = pd.DataFrame({
    "Масштабування": ["Ні", "StandardScaler"],
    "R²": [r2_no_scaling, r2_scaled]
})

print(scaling_results)