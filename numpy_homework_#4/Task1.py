import pandas as pd

data_books = {
    "Назва": ["Книга1", "Книга2", "Книга3", "Книга4", "Книга5", "Книга6"],
    "Автор": ["Автор1", "Автор2", "Автор3", "Автор4", "Автор5", "Автор6"],
    "Рік видання": [2010, 2018, 2020, 2015, 2022, 2017],
    "Ціна": [200, 350, 500, 150, 600, 300]
}

df_books = pd.DataFrame(data_books)

print("Весь DataFrame:")
print(df_books)

avg_price = df_books["Ціна"].mean()
print("\nСередня ціна:", avg_price)

print("\nКниги після 2015 року:")
print(df_books[df_books["Рік видання"] > 2015])

print("\nСортування за ціною (зростання):")
print(df_books.sort_values(by="Ціна"))