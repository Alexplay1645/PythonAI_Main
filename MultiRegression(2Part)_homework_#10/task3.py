import numpy as np
import pandas as pd
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer

X_missing = X.copy()

rng = np.random.RandomState(42)
missing_mask = rng.rand(*X_missing[:, 0].shape) < 0.1
X_missing[missing_mask, 0] = np.nan

model_clean = LinearRegression()
model_clean.fit(X_train, y_train)
r2_clean = model_clean.score(X_test, y_test)

imputer = SimpleImputer(strategy="mean")
X_imputed = imputer.fit_transform(X_missing)

X_train_i, X_test_i, y_train_i, y_test_i = train_test_split(
    X_imputed, y, test_size=0.2, random_state=42
)

model_imputed = LinearRegression()
model_imputed.fit(X_train_i, y_train_i)
r2_imputed = model_imputed.score(X_test_i, y_test_i)

impute_results = pd.DataFrame({
    "Сценарій": ["Без пропусків", "Після імпутації"],
    "R²": [r2_clean, r2_imputed]
})

print(impute_results)