import pandas as pd

data_tickets = {
    "Фільм": ["A", "A", "A", "B", "B", "B", "C", "C", "C", "D", "D", "D"],
    "Місто": ["Київ", "Львів", "Одеса", "Київ", "Львів", "Одеса", "Київ", "Львів", "Одеса", "Київ", "Львів", "Одеса"],
    "Продані квитки": [100, 80, 90, 120, 110, 100, 70, 60, 50, 200, 180, 160]
}

df_tickets = pd.DataFrame(data_tickets)

print(df_tickets)

print("\nКвитки по фільмах:")
print(df_tickets.groupby("Фільм")["Продані квитки"].sum())

print("\nКвитки по містах:")
print(df_tickets.groupby("Місто")["Продані квитки"].sum())

print("\nНайпопулярніший фільм:")
print(df_tickets.groupby("Фільм")["Продані квитки"].sum().idxmax())