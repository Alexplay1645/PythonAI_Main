import numpy as np

def run():

    n = int(input("Довжина масивів: "))

    arr1 = np.random.randint(0, 11, n)
    arr2 = np.random.randint(10, 21, n)

    print("Масив 1:", arr1)
    print("Масив 2:", arr2)

    merged = np.concatenate((arr1, arr2))
    print("Об'єднаний масив:", merged)

    sum_arr = arr1 + arr2
    diff_arr = arr1 - arr2

    print("Поелементна сума:", sum_arr)
    print("Поелементна різниця:", diff_arr)