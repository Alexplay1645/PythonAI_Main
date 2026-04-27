import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

population = np.random.exponential(scale=2, size=50000)

def sample_means(n, num_samples=1000):
    means = []
    for _ in range(num_samples):
        sample = np.random.choice(population, size=n)
        means.append(sample.mean())
    return means

means_10 = sample_means(10)
means_50 = sample_means(50)

plt.hist(means_10, bins=30)
plt.title("Sample means (n=10)")
plt.show()

plt.hist(means_50, bins=30)
plt.title("Sample means (n=50)")
plt.show()

print("n=10 -> mean:", np.mean(means_10), "std:", np.std(means_10))
print("n=50 -> mean:", np.mean(means_50), "std:", np.std(means_50))