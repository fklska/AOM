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

# Размер популяции — 10, поколения идут последовательно
pop_size = 10
df["generation"] = df.index // pop_size

# Для удобства выделим лучшие особи по поколениям
best_per_gen = df.loc[df.groupby("generation")["roc_auc"].idxmax()]

# Статистики по поколениям (игнорируем NaN)
gen_stats = df.groupby("generation")["roc_auc"].agg(["max", "mean", "min"]).reset_index()

plt.figure(figsize=(10, 5))
plt.plot(gen_stats["generation"], gen_stats["max"], marker="o", label="Лучший AUC")
plt.plot(gen_stats["generation"], gen_stats["mean"], marker="s", label="Средний AUC")
plt.plot(gen_stats["generation"], gen_stats["min"], marker="v", label="Худший AUC")
plt.xlabel("Поколение")
plt.ylabel("ROC-AUC")
plt.title("Сходимость генетического алгоритма")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("ga_convergence.png", dpi=150)
plt.show()

plt.figure(figsize=(10, 5))
sns.boxplot(data=df.dropna(subset=["roc_auc"]), x="generation", y="roc_auc")
sns.stripplot(data=df.dropna(subset=["roc_auc"]), x="generation", y="roc_auc", 
              color="red", alpha=0.3, size=3)  # точки-выбросы
plt.title("Распределение ROC-AUC по поколениям")
plt.xlabel("Поколение")
plt.ylabel("ROC-AUC")
plt.ylim(0.5, 0.75)  # подрежем, чтобы были видны различия
plt.tight_layout()
plt.savefig("ga_boxplot.png", dpi=150)
plt.show()

# Функция для подсчёта частоты гена в поколении
def gene_freq(column):
    freq = df.groupby(["generation", column]).size().unstack(fill_value=0)
    freq = freq.div(freq.sum(axis=1), axis=0)  # нормируем в доли
    return freq

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, gene, title in zip(axes, ["role", "instruct", "constraint"],
                           ["Роли", "Инструкции", "Ограничения"]):
    freq = gene_freq(gene)
    sns.heatmap(freq, annot=True, fmt=".1f", cmap="YlOrRd", ax=ax,
                cbar_kws={'label': 'Доля в поколении'})
    ax.set_title(title)
    ax.set_xlabel("Индекс гена")
    ax.set_ylabel("Поколение")
plt.tight_layout()
plt.savefig("ga_gene_frequency.png", dpi=150)
plt.show()

plt.figure(figsize=(8, 4))
plt.scatter(best_per_gen["generation"], best_per_gen["roc_auc"], s=100, c="red", zorder=5)
plt.plot(best_per_gen["generation"], best_per_gen["roc_auc"], "--", alpha=0.5)
for _, row in best_per_gen.iterrows():
    plt.annotate(f"[{row['role']},{row['instruct']},{row['constraint']}]", 
                 (row["generation"], row["roc_auc"]),
                 textcoords="offset points", xytext=(0,10), ha="center")
plt.xlabel("Поколение")
plt.ylabel("Лучший AUC")
plt.title("Траектория лучшей особи")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("ga_best_individual.png", dpi=150)
plt.show()