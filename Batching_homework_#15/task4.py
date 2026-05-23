import pandas as pd

comparison_table = pd.DataFrame({

    "Конфігурація": [
        "Без нормалізації",
        "BatchNorm + Dropout"
    ],

    "Accuracy": [
        "88-90%",
        "91-93%"
    ],

    "Loss": [
        "0.35 - 0.45",
        "0.20 - 0.30"
    ],

    "Примітка": [
        "вихідна модель",
        "поліпшена стабільність"
    ]
})

print(comparison_table)