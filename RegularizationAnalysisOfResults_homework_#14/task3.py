from sklearn.metrics import mean_absolute_error, r2_score

X_train_tensor = torch.FloatTensor(X_train_scaled)
X_test_tensor = torch.FloatTensor(X_test_scaled)

y_train_tensor = torch.FloatTensor(y_train.values).view(-1, 1)
y_test_tensor = torch.FloatTensor(y_test.values).view(-1, 1)

model_no_reg = nn.Sequential(
    nn.Linear(input_size, 64),
    nn.ReLU(),

    nn.Linear(64, 32),
    nn.ReLU(),

    nn.Linear(32, 16),
    nn.ReLU(),

    nn.Linear(16, 1)
)

criterion = nn.MSELoss()

optimizer_no_reg = optim.Adam(
    model_no_reg.parameters(),
    lr=0.001
)

epochs = 50

for epoch in range(epochs):

    predictions = model_no_reg(X_train_tensor)

    loss = criterion(predictions, y_train_tensor)

    optimizer_no_reg.zero_grad()
    loss.backward()
    optimizer_no_reg.step()

with torch.no_grad():
    y_pred_no_reg = model_no_reg(X_test_tensor).numpy()

mae_no_reg = mean_absolute_error(y_test, y_pred_no_reg)
r2_no_reg = r2_score(y_test, y_pred_no_reg)

model_l2 = nn.Sequential(
    nn.Linear(input_size, 64),
    nn.ReLU(),

    nn.Linear(64, 32),
    nn.ReLU(),

    nn.Linear(32, 16),
    nn.ReLU(),

    nn.Linear(16, 1)
)

optimizer_l2 = optim.Adam(
    model_l2.parameters(),
    lr=0.001,
    weight_decay=1e-4
)

for epoch in range(epochs):

    predictions = model_l2(X_train_tensor)

    loss = criterion(predictions, y_train_tensor)

    optimizer_l2.zero_grad()
    loss.backward()
    optimizer_l2.step()

with torch.no_grad():
    y_pred_l2 = model_l2(X_test_tensor).numpy()

mae_l2 = mean_absolute_error(y_test, y_pred_l2)
r2_l2 = r2_score(y_test, y_pred_l2)

print("Конфігурація\t\t\tMAE\t\tR²")

print(f"Без регуляризації\t\t{mae_no_reg:.2f}\t\t{r2_no_reg:.2f}")

print(f"З регуляризацією (L2)\t\t{mae_l2:.2f}\t\t{r2_l2:.2f}")