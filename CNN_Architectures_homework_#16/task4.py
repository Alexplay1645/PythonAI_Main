import matplotlib.pyplot as plt
import numpy as np
import random

class_names = ["NORMAL", "PNEUMONIA"]

model.eval()

indices = random.sample(
    range(len(test_dataset)),
    8
)

fig, axes = plt.subplots(
    2,
    4,
    figsize=(14, 8)
)

axes = axes.flatten()

with torch.no_grad():

    for i, idx in enumerate(indices):

        image, label = test_dataset[idx]

        input_image = image.unsqueeze(0).to(device)

        output = model(input_image)

        _, pred = torch.max(output, 1)

        img = image.numpy().transpose((1,2,0))

        mean = np.array([0.485,0.456,0.406])
        std = np.array([0.229,0.224,0.225])

        img = std * img + mean
        img = np.clip(img, 0, 1)

        axes[i].imshow(img)

        axes[i].set_title(
            f"Pred: {class_names[pred.item()]}\n"
            f"Real: {class_names[label]}"
        )

        axes[i].axis("off")

plt.tight_layout()

plt.savefig(
    "pneumonia_predictions.png"
)

plt.show()

print(
    "Visualization saved as pneumonia_predictions.png"
)