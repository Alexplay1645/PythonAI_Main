import matplotlib.pyplot as plt
import numpy as np

days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

temperature = [20, 22, 19, 23, 25, 24, 21]
humidity = [60, 65, 55, 70, 75, 72, 68]

plt.figure()

plt.plot(days, temperature, marker='o', label='Temperature')
plt.plot(days, humidity, marker='o', label='Humidity')

plt.xlabel('Days')
plt.ylabel('Values')
plt.title('Temperature and Humidity')

plt.xticks(rotation=45)
plt.legend()

plt.show()