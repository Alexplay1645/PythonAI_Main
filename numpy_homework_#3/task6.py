import numpy as np

def run():

    rows = int(input("Кількість рядків: "))
    cols = int(input("Кількість стовпців: "))

    matrix = np.arange(rows * cols).reshape(rows, cols)
    print("Початкова матриця:\n", matrix)

    new_rows = int(input("Нові рядки: "))
    new_cols = int(input("Нові стовпці: "))

    if rows * cols != new_rows * new_cols:
        print("Неможливо змінити розмір матриці (кількість елементів не співпадає)")
        return

    new_matrix = matrix.reshape(new_rows, new_cols)
    print("Нова матриця:\n", new_matrix)

    min_vals = np.min(new_matrix, axis=1)
    max_vals = np.max(new_matrix, axis=1)

    print("Мінімуми рядків:", min_vals)
    print("Максимуми рядків:", max_vals)

    total_sum = np.sum(new_matrix)
    print("Сума всіх елементів:", total_sum)