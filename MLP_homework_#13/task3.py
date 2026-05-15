import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

X, y = make_classification(
    n_samples=1000,
    n_features=10,
    n_informative=6,
    n_redundant=2,
    random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train_tensor = torch.FloatTensor(X_train)
X_test_tensor = torch.FloatTensor(X_test)

y_train_tensor = torch.FloatTensor(y_train).view(-1, 1)
y_test_tensor = torch.FloatTensor(y_test).view(-1, 1)

input_size = X_train.shape[1]

model = nn.Sequential(
    nn.Linear(input_size, 16),
    nn.ReLU(),

    nn.Linear(16, 8),
    nn.ReLU(),

    nn.Linear(8, 1),
    nn.Sigmoid()
)

criterion = nn.BCELoss()

optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 30


for epoch in range(epochs):

    model.train()

    outputs = model(X_train_tensor)

    loss = criterion(outputs, y_train_tensor)

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    predicted = (outputs >= 0.5).float()

    accuracy = accuracy_score(
        y_train_tensor.detach().numpy(),
        predicted.detach().numpy()
    )

    print(
        f"Епоха [{epoch + 1}/{epochs}] | "
        f"Loss: {loss.item():.4f} | "
        f"Accuracy: {accuracy:.4f}"
    )

print("\n================================")
print("ПІДСУМКОВІ МЕТРИКИ")
print("================================")

print(f"Loss: {loss.item():.4f}")
print(f"Accuracy: {accuracy:.4f}")