import numpy as np

low = int(input("Введіть нижню межу: "))
high = int(input("Введіть верхню межу: "))

matrix = np.random.randint(low, high + 1, (5, 5))

print("Вихідна матриця:")
print(matrix)

diagonal = np.diag(matrix)
print("Головна діагональ:", diagonal)

diag_sum = np.sum(diagonal)
print("Сума діагоналі:", diag_sum)

result = matrix.copy()

for i in range(5):
    for j in range(5):
        if j > i:
            result[i][j] = 0

print("Матриця після змін:")
print(result)