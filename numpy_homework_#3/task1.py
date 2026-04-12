import numpy as np

def run():

    arr = np.random.randint(0, 51, 20)
    print("Масив:", arr)

    threshold = int(input("Введіть порогове значення: "))

    count = np.sum(arr > threshold)
    print("Кількість елементів більше порогу:", count)

    max_val = np.max(arr)
    max_index = np.argmax(arr)
    print("Максимум:", max_val)
    print("Позиція:", max_index)

    sorted_arr = np.sort(arr)[::-1]
    print("Відсортований масив:", sorted_arr)