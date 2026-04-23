import matplotlib.pyplot as plt
import numpy as np

ages = np.random.randint(18, 60, 100)

mean_age = np.mean(ages)

plt.figure()
plt.hist(ages, bins=10)

plt.axvline(mean_age)

plt.xlabel('Age')
plt.ylabel('Frequency')
plt.title('Age Distribution')

plt.show()