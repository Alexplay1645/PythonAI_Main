import torch
import torch.nn as nn
import torch.optim as optim

input_size = X_train_scaled.shape[1]

model = nn.Sequential(
    nn.Linear(input_size, 64),
    nn.ReLU(),

    nn.Linear(64, 32),
    nn.ReLU(),

    nn.Linear(32, 16),
    nn.ReLU(),

    nn.Linear(16, 1)
)

print(model)

criterion = nn.MSELoss()

optimizer = optim.Adam(model.parameters(), lr=0.001)