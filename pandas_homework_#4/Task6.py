import pandas as pd

clients = {
    "ID": [1, 2, 3, 4, 5],
    "Ім'я": ["Іван", "Марія", "Петро", "Олена", "Андрій"],
    "Місто": ["Київ", "Львів", "Одеса", "Харків", "Дніпро"]
}

orders = {
    "ID клієнта": [1, 2, 1, 3, 5],
    "Замовлення": ["A", "B", "C", "D", "E"],
    "Вартість": [500, 700, 300, 400, 900]
}

df_clients = pd.DataFrame(clients)
df_orders = pd.DataFrame(orders)

print(df_clients)
print(df_orders)

merged = pd.merge(df_clients, df_orders, left_on="ID", right_on="ID клієнта")

print("\nОб'єднаний DataFrame:")
print(merged)

result = merged.groupby("Ім'я")["Вартість"].sum().reset_index()

print("\nСума замовлень по клієнтам:")
print(result)