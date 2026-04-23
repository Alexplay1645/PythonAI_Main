import matplotlib.pyplot as plt
import numpy as np

time = np.arange(10)

conversion = np.random.rand(10)
retention = np.random.rand(10)
avg_check = np.random.rand(10)
orders = np.random.randint(10, 100, 10)

fig, axs = plt.subplots(2, 2)

axs[0, 0].plot(time, conversion)
axs[0, 0].set_title('Conversion')

axs[0, 1].plot(time, retention)
axs[0, 1].set_title('Retention')

axs[1, 0].plot(time, avg_check)
axs[1, 0].set_title('Avg Check')

axs[1, 1].plot(time, orders)
axs[1, 1].set_title('Orders')

fig.suptitle('Product Metrics')

plt.show()