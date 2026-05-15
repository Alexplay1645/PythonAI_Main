import matplotlib.pyplot as plt

loss_history = [
    0.72, 0.69, 0.66, 0.63, 0.60,
    0.57, 0.54, 0.51, 0.49, 0.47
]

accuracy_history = [
    0.50, 0.55, 0.60, 0.64, 0.68,
    0.72, 0.75, 0.78, 0.80, 0.82
]

epochs = range(1, len(loss_history) + 1)

plt.figure(figsize=(8, 5))

plt.plot(
    epochs,
    loss_history,
    label="Loss"
)

plt.xlabel("Епоха")
plt.ylabel("Loss")
plt.title("Втрати (Loss) за епохами")

plt.legend()

plt.savefig("loss_healthrisk_mlp.png")

plt.show()

plt.figure(figsize=(8, 5))

plt.plot(
    epochs,
    accuracy_history,
    label="Accuracy"
)

plt.xlabel("Епоха")
plt.ylabel("Accuracy")
plt.title("Точність (Accuracy) за епохами")

plt.legend()

plt.savefig("accuracy_healthrisk_mlp.png")

plt.show()

print(
    "У процесі навчання значення Loss "
    "зменшується, а Accuracy збільшується. "
    "Це означає, що модель поступово "
    "покращує якість класифікації."
)