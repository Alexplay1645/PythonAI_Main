import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.metrics import r2_score
from sklearn.linear_model import Ridge

linear_model = LinearRegression()

linear_model.fit(X_train, y_train)

y_pred_linear = linear_model.predict(X_test)

linear_r2 = r2_score(y_test, y_pred_linear)

linear_nonzero = sum(linear_model.coef_ != 0)

ridge_model = Ridge(alpha=1.0)

ridge_model.fit(X_train, y_train)

y_pred_ridge = ridge_model.predict(X_test)

ridge_r2 = r2_score(y_test, y_pred_ridge)

ridge_nonzero = sum(ridge_model.coef_ != 0)

results_ridge = pd.DataFrame({
    'Модель': [
        'LinearRegression',
        'Ridge(alpha=1.0)'
    ],
    'Test R²': [
        round(linear_r2, 4),
        round(ridge_r2, 4)
    ],
    'Кількість ненульових коеф.': [
        linear_nonzero,
        ridge_nonzero
    ]
})

print(results_ridge)