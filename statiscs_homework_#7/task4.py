import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dates = pd.date_range(start="2024-01-01", periods=90)

sales = pd.DataFrame({
    "date": dates,
    "sales": np.random.randint(50, 200, size=90)
})

window = 7
sales["rolling_mean"] = sales["sales"].rolling(window).mean()
sales["rolling_std"] = sales["sales"].rolling(window).std()

print(sales.head(10))

plt.plot(sales["date"], sales["sales"], label="Sales")
plt.plot(sales["date"], sales["rolling_mean"], label="Rolling Mean")
plt.legend()
plt.xticks(rotation=45)
plt.show()

plt.plot(sales["date"], sales["rolling_std"])
plt.title("Rolling Std Dev")
plt.xticks(rotation=45)
plt.show()