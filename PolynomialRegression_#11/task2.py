import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

results = []

for degree in [1, 2, 3]:

    poly = PolynomialFeatures(degree=degree)

    X_poly = poly.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_poly,
        y,
        test_size=0.2,
        random_state=42
    )
    
    model = LinearRegression()

    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    results.append([
        degree,
        round(train_r2, 4),
        round(test_r2, 4)
    ])

results_df = pd.DataFrame(
    results,
    columns=[
        "Ступінь полінома",
        "R² train",
        "R² test"
    ]
)

print(results_df)