import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

A = pd.DataFrame({
    "group": "A",
    "converted": np.random.binomial(1, 0.10, 120)
})

B = pd.DataFrame({
    "group": "B",
    "converted": np.random.binomial(1, 0.13, 120)
})

df = pd.concat([A, B])

conversion = df.groupby("group")["converted"].mean()
print("Conversion:\n", conversion)

abs_diff = conversion["B"] - conversion["A"]
rel_change = abs_diff / conversion["A"]

print("Absolute difference:", abs_diff)
print("Relative change:", rel_change)

def ci(p, n):
    se = np.sqrt(p * (1 - p) / n)
    return 1.96 * se

n_A = len(A)
n_B = len(B)

ci_A = ci(conversion["A"], n_A)
ci_B = ci(conversion["B"], n_B)

plt.bar(["A", "B"], conversion, yerr=[ci_A, ci_B], capsize=5)
plt.title("Conversion Rate with 95% CI")
plt.show()