import pandas as pd

df_orders = pd.read_csv("orders.csv")

print("Перші 10 рядків:")
print(df_orders.head(10))

print("\nКількість замовлень по клієнтам:")
print(df_orders["Клієнт"].value_counts())

print("\nМаксимальна сума:", df_orders["Сума"].max())
print("Мінімальна сума:", df_orders["Сума"].min())

print("\nЗагальна сума:")
print(df_orders["Сума"].sum())