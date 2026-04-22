import pandas as pd

data_workers = {
    "Ім'я": ["Іван", "Марія", "Петро", "Олена", "Іван", "Марія", "Петро", "Олена"],
    "Проект": ["A", "A", "B", "B", "C", "C", "A", "B"],
    "Години": [10, 12, 8, 15, 20, 10, 5, 7]
}

df_workers = pd.DataFrame(data_workers)

print(df_workers)

print("\nГодини по співробітникам:")
print(df_workers.groupby("Ім'я")["Години"].sum())

print("\nГодини по проектам:")
print(df_workers.groupby("Проект")["Години"].sum())

print("\nНайбільше годин:")
print(df_workers.groupby("Ім'я")["Години"].sum().idxmax())