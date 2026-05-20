import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))

plt.scatter(y_test, y_pred_l2)

plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()]
)

plt.xlabel("Реальні значення")
plt.ylabel("Передбачені значення")

plt.title("Real vs Predicted Values")

plt.savefig("diabetes_healthrisk_analysis.png")

plt.show()