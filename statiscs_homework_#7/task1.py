import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

dates = pd.date_range(start="2024-01-01", periods=30)

data = pd.DataFrame({
    "date": dates,
    "users": np.random.randint(100, 500, size=30),
    "sessions": np.random.randint(150, 800, size=30),
    "revenue": np.random.randint(1000, 5000, size=30)
})

print(data.head())

correlation_matrix = data[["users", "sessions", "revenue"]].corr()

print("Кореляційна матриця:")
print(correlation_matrix)

plt.figure()
plt.scatter(data["users"], data["sessions"])
plt.xlabel("Users")
plt.ylabel("Sessions")
plt.title("Users vs Sessions")
plt.show()

plt.figure()
plt.scatter(data["users"], data["revenue"])
plt.xlabel("Users")
plt.ylabel("Revenue")
plt.title("Users vs Revenue")
plt.show()

plt.figure()
plt.scatter(data["sessions"], data["revenue"])
plt.xlabel("Sessions")
plt.ylabel("Revenue")
plt.title("Sessions vs Revenue")
plt.show()

plt.figure()
plt.plot(data["date"], data["revenue"])
plt.xlabel("Date")
plt.ylabel("Revenue")
plt.title("Revenue over Time")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()