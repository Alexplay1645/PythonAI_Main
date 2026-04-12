import numpy as np

def run():

    low = int(input("Нижня межа: "))
    high = int(input("Верхня межа: "))

    matrix = np.random.randint(low, high + 1, (5, 5))
    print("Матриця:\n", matrix)

    diag = np.diag(matrix)
    print("Головна діагональ:", diag)

    print("Сума діагоналі:", np.sum(diag))

    matrix[np.triu_indices(5, 1)] = 0
    print("Матриця після змін:\n", matrix)