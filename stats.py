import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Читаем логи
with open("evo_logs3.json", "r", encoding="utf-8") as f:
    logs = [json.loads(line) for line in f if line.strip()]

df = pd.DataFrame(logs)

# Заменяем NaN на None, а 0 на NaN (чтобы исключить «битые» запуски из средних)
df["roc_auc"] = df["roc_auc"]  # 0 считаем как неудачный запуск

# display(df.head())

# # Размер популяции — 10, поколения идут последовательно
# pop_size = 10
# df["generation"] = df.index // pop_size

# # Для удобства выделим лучшие особи по поколениям
# best_per_gen = df.loc[df.groupby("generation")["roc_auc"].idxmax()]

# # Статистики по поколениям (игнорируем NaN)
# gen_stats = df.groupby("generation")["roc_auc"].agg(["max", "mean", "min"]).reset_index()