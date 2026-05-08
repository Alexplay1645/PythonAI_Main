import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.metrics import r2_score

housing = fetch_california_housing(as_frame=True)

X = housing.data[['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms']]

y = housing.target

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

linear_model = LinearRegression()

linear_model.fit(X_train, y_train)

y_pred_linear = linear_model.predict(X_test)

linear_r2 = r2_score(y_test, y_pred_linear)

linear_coef = linear_model.coef_

lasso_model = Lasso(alpha=0.1)

lasso_model.fit(X_train, y_train)

y_pred_lasso = lasso_model.predict(X_test)

lasso_r2 = r2_score(y_test, y_pred_lasso)

lasso_coef = lasso_model.coef_

zero_coef_count = sum(lasso_coef == 0)

results_lasso = pd.DataFrame({
    'Модель': ['LinearRegression', 'Lasso(alpha=0.1)'],
    'Test R²': [
        round(linear_r2, 4),
        round(lasso_r2, 4)
    ],
    'Кількість нульових коеф.': [
        sum(linear_coef == 0),
        zero_coef_count
    ]
})

print(results_lasso)

print("\nКоефіцієнти LinearRegression:")
print(linear_coef)

print("\nКоефіцієнти Lasso:")
print(lasso_coef)
