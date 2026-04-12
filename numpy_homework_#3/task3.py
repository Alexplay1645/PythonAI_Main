import numpy as np

def run():

    start = int(input("Початок діапазону: "))
    end = int(input("Кінець діапазону: "))

    sequence = np.arange(start, end + 1)

    if len(sequence) < 30:
        print("Недостатньо чисел для створення матриці 6x5 (потрібно мінімум 30)")
        return

    sequence = sequence[:30]

    matrix = sequence.reshape(6, 5)
    print("Матриця:\n", matrix)

    row_sums = np.sum(matrix, axis=1)
    print("Суми рядків:", row_sums)

    col_max = np.max(matrix, axis=0)
    print("Максимуми стовпців:", col_max)