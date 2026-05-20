import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Читаем JSON Lines
logs = []
with open("logs.json", "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            logs.append(json.loads(line))

# Превращаем в DataFrame
df = pd.DataFrame(logs)

# Проверим размер
print(f"Всего записей: {len(df)}")

# Добавляем номер поколения: первые 10 -> поколение 0, следующие 10 -> 1 и т.д.
generation_size = 10
df["generation"] = df.index // generation_size

# Если последняя часть не кратна 10, всё равно маркируем правильно
# (у нас 40 записей, всё отлично)