import pandas as pd
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

model_no_scale = LinearRegression()
model_no_scale.fit(X_train, y_train)
y_pred_no_scale = model_no_scale.predict(X_test)
r2_no_scale = r2_score(y_test, y_pred_no_scale)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model_scaled = LinearRegression()
model_scaled.fit(X_train_scaled, y_train)
y_pred_scaled = model_scaled.predict(X_test_scaled)
r2_scaled = r2_score(y_test, y_pred_scaled)

df_compare = pd.DataFrame({
    "Масштабування": ["Ні", "StandardScaler"],
    "R²": [r2_no_scale, r2_scaled]
})

print("\nПорівняння:")
print(df_compare)