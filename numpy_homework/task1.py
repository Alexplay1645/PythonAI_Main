import numpy as np

arr = np.random.randint(0, 51, 20)

print("Вихідний масив:")
print(arr)

threshold = int(input("Введіть порогове значення: "))

count = np.sum(arr > threshold)
print("Кількість елементів більше порогу:", count)

max_value = np.max(arr)
max_index = np.argmax(arr)

print("Максимальне значення:", max_value)
print("Позиція першого входження:", max_index)

sorted_arr = np.sort(arr)[::-1]

print("Відсортований масив:")
print(sorted_arr)