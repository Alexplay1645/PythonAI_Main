import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

diabetes = load_diabetes()

X = diabetes.data
y = diabetes.target

print("Форма матриці ознак X:", X.shape)
print("Форма цільової змінної y:", y.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nПісля розділення даних:")

print("Форма X_train:", X_train_scaled.shape)
print("Форма X_test:", X_test_scaled.shape)

print("Форма y_train:", y_train.shape)
print("Форма y_test:", y_test.shape)