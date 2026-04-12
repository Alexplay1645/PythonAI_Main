import numpy as np

def run():

    low = int(input("Нижня межа: "))
    high = int(input("Верхня межа: "))

    arr = np.random.randint(low, high + 1, 15)
    print("Масив:", arr)

    negatives = arr[arr < 0]
    print("Від'ємні елементи:", negatives)

    modified = arr.copy()
    modified[modified < 0] = 0
    print("Змінений масив:", modified)

    zeros_count = np.sum(modified == 0)
    print("Кількість нулів:", zeros_count)