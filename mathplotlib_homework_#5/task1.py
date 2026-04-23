import matplotlib.pyplot as plt
import numpy as np

months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']

plan = [100, 120, 130, 150, 170, 200]
fact = [90, 110, 140, 140, 180, 210]

plt.figure()
plt.plot(months, plan, marker='o', label='Plan')
plt.plot(months, fact, marker='o', label='Fact')

plt.xlabel('Months')
plt.ylabel('Sales')
plt.title('Plan vs Fact Sales')
plt.legend()

plt.show()