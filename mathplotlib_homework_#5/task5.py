import matplotlib.pyplot as plt
import numpy as np

hours = list(range(24))
load = np.random.randint(20, 80, 24)

plt.figure()

plt.plot(hours, load)
plt.fill_between(hours, load)

plt.xlabel('Hours')
plt.ylabel('Server Load')
plt.title('Server Load per Day')

plt.grid()

plt.show()