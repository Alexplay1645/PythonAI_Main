import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

data = fetch_california_housing(as_frame=True)
df = data.frame.sample(n=5000, random_state=42)

X = df[['MedInc']]
y = df['MedHouseVal']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)

result1 = pd.DataFrame({
    "Модель": ["LinearRegression"],
    "Ознака": ["MedInc"],
    "R²": [r2]
})

print(result1)