import matplotlib.pyplot as plt

class BasicCNN(nn.Module):
    def __init__(self):
        super(BasicCNN, self).__init__()

        self.conv_layers = nn.Sequential(

            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

def train_model(model):

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    loss_history = []

    for epoch in range(10):

        model.train()

        running_loss = 0.0

        for images, labels in train_loader:

            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)

        loss_history.append(epoch_loss)

        print(f"Epoch {epoch+1}, Loss: {epoch_loss:.4f}")

    return loss_history

basic_model = BasicCNN().to(device)

improved_model = CNN_BatchNorm().to(device)

print("Навчання базової моделі...")
basic_losses = train_model(basic_model)

print("Навчання покращеної моделі...")
improved_losses = train_model(improved_model)

plt.plot(basic_losses, label="Без нормалізації")

plt.plot(improved_losses, label="BatchNorm + Dropout")

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.title("Порівняння моделей")

plt.legend()

plt.show()