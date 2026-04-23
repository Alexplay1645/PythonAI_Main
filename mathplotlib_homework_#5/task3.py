import matplotlib.pyplot as plt
import numpy as np

group1 = np.random.randint(60, 100, 30)
group2 = np.random.randint(50, 95, 30)
group3 = np.random.randint(40, 90, 30)

plt.figure()
plt.boxplot([group1, group2, group3])

plt.xticks([1, 2, 3], ['Group 1', 'Group 2', 'Group 3'])

plt.xlabel('Groups')
plt.ylabel('Scores')
plt.title('Exam Results Comparison')

plt.show()