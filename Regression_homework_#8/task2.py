import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

X1 = df[['MedInc']]

X2 = df[['MedInc', 'AveRooms']]

X1_train, X1_test, y_train, y_test = train_test_split(
    X1, y, test_size=0.2, random_state=42
)

X2_train, X2_test, _, _ = train_test_split(
    X2, y, test_size=0.2, random_state=42
)

model1 = LinearRegression()
model1.fit(X1_train, y_train)
r2_1 = r2_score(y_test, model1.predict(X1_test))

model2 = LinearRegression()
model2.fit(X2_train, y_train)
r2_2 = r2_score(y_test, model2.predict(X2_test))

result2 = pd.DataFrame({
    "Модель": ["LinearRegression", "LinearRegression"],
    "Ознаки": ["MedInc", "MedInc, AveRooms"],
    "R²": [r2_1, r2_2]
})

print(result2)