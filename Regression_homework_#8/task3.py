import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

from sklearn.metrics import mean_absolute_error, mean_squared_error

model = LinearRegression()
model.fit(X2_train, y_train)

y_pred = model.predict(X2_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

result3 = pd.DataFrame({
    "Метрика": ["MAE", "MSE", "RMSE", "R²"],
    "Значення": [mae, mse, rmse, r2]
})

print(result3)