import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

housing = fetch_california_housing()

X = pd.DataFrame(
    housing.data,
    columns=housing.feature_names
)

y = housing.target

X = X[['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms']]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

tree_model = DecisionTreeRegressor(
    max_depth=4,
    random_state=42
)

tree_model.fit(X_train, y_train)

y_pred_tree = tree_model.predict(X_test)

r2_tree = r2_score(y_test, y_pred_tree)

forest_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

forest_model.fit(X_train, y_train)

y_pred_forest = forest_model.predict(X_test)

r2_forest = r2_score(y_test, y_pred_forest)

results = pd.DataFrame({
    'Model': [
        'Decision Tree',
        'Random Forest'
    ],
    'R² Score': [
        r2_tree,
        r2_forest
    ]
})

print(results)