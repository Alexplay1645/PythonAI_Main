import pandas as pd
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

heart = fetch_openml(
    name='heart-disease',
    version=1,
    as_frame=True
)

X = heart.data
y = heart.target

print("Перші 5 рядків:")
print(X.head())

print("\nРозмір:")
print(X.shape)

print("\nТипи даних:")
print(X.dtypes)

print("\nПропущені значення:")
print(X.isnull().sum())

numeric_columns = X.select_dtypes(
    include=['int64', 'float64']
).columns

for col in numeric_columns:
    X[col] = X[col].fillna(
        X[col].median()
    )

categorical_columns = X.select_dtypes(
    include=['object', 'category']
).columns

for col in categorical_columns:
    X[col] = X[col].fillna(
        X[col].mode()[0]
    )

print("\nБаланс класів:")
print(y.value_counts())

X = pd.get_dummies(
    X,
    drop_first=True
)


try:
    y = y.map({
        'present': 1,
        'absence': 0,
        '1': 1,
        '0': 0
    })
except:
    y = y.astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(
    X_train
)

X_test_scaled = scaler.transform(
    X_test
)

print("\nФорма train:")
print(X_train_scaled.shape)

print("\nФорма test:")
print(X_test_scaled.shape)

print("\nБаланс train:")
print(y_train.value_counts(normalize=True))

print("\nБаланс test:")
print(y_test.value_counts(normalize=True))

print("\nДані готові для MLP.")