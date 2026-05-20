import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ======================== 1. Данные ========================
# --- Начальные 4 промпта (примеры, замените на свои) ---
initial_data = [
    {"prompt": "init_1", "roc_auc": 0.0},
    {"prompt": "init_2", "roc_auc": 0.584295},
    {"prompt": "init_3", "roc_auc": 0.637306},
    {"prompt": "init_4", "roc_auc": 0.655533},
]
df_init = pd.DataFrame(initial_data)
df_init["x"] = 0   # все в начальной точке

# --- Логи LLM-optimizer (ваши 7 записей) ---
llm_logs = [
    {"roc_auc": 0.697965},
    {"roc_auc": 0.729046},
    {"roc_auc": 0.725949},
    {"roc_auc": 0.725949},
    {"roc_auc": 0},
    {"roc_auc": 0},
    {"roc_auc": 0.714678},
    {"roc_auc": 0.714678},
    {"roc_auc": 0},
    {"roc_auc": 0.725949},  
]
df_llm = pd.DataFrame(llm_logs)
df_llm["x"] = range(1, len(df_llm) + 1)

# Глобальный максимум дискретного пространства
global_max = 0.741475

# ======================== 2. Визуализация ========================
plt.figure(figsize=(12, 6))

# 2.1 Начальные промпты (синие квадраты)
plt.scatter(df_init["x"], df_init["roc_auc"], 
            color='royalblue', s=120, marker='s', 
            edgecolors='black', linewidth=0.5, label='Начальные промпты (4 шт.)')

# 2.2 Итерации LLM с разделением на «нормальные» и «нерабочие»
# Нормальные (AUC > 0)
df_ok = df_llm[df_llm["roc_auc"] > 0]
sc = plt.scatter(df_ok["x"], df_ok["roc_auc"], 
                 c=df_ok["roc_auc"], cmap='spring', s=150, marker='o', 
                 edgecolors='black', linewidth=0.5, vmin=0.65, vmax=0.75)
plt.colorbar(sc, label='ROC-AUC (нормальные запуски)')

# Нерабочие (AUC = 0) – красные кружки с чёрной окантовкой
df_bad = df_llm[df_llm["roc_auc"] == 0]
if not df_bad.empty:
    plt.scatter(df_bad["x"], df_bad["roc_auc"], 
                color='red', s=150, marker='o', 
                edgecolors='black', linewidth=1.5, label='Нерабочий код (AUC=0)')

# 2.3 Глобальный максимум
plt.axhline(y=global_max, color='green', linestyle='--', linewidth=2, 
            label=f'Глобальный максимум (дискрет) = {global_max:.4f}')

# Оформление
plt.xlabel("Итерация LLM-optimizer (0 — начальные)")
plt.ylabel("ROC-AUC")
plt.title("LLM-as-Optimizer")
plt.xticks(range(0, len(df_llm) + 1))
plt.grid(axis='y', alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("llm_scatter_plot.png", dpi=150)
plt.show()