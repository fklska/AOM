from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d import Axes3D
from plotly import colors  

LOG_PATH = Path("logs_bf.json")
PROMPTS_PATH = Path("prompts.csv")
OUT_DIR = Path("pics_new")


def load_logs(path: Path) -> pd.DataFrame:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    df = pd.DataFrame(rows)
    # ожидаем столбцы role, instruct, constraint, roc_auc
    df["role"] = df["role"].astype(int)
    df["instruct"] = df["instruct"].astype(int)
    df["constraint"] = df["constraint"].astype(int)
    return df


def load_prompt_labels(path: Path, max_len: int = 30) -> tuple[list[str], list[str], list[str]]:
    """Считать тексты ролей / инструкций / ограничений и сделать короткие подписи для осей."""
    prompts = pd.read_csv(path, sep=";")

    def make_labels(series: pd.Series) -> list[str]:
        labels: list[str] = []
        for idx, text in enumerate(series.tolist()):
            text = str(text).replace("\n", " ").strip()
            if len(text) > max_len:
                text = text[: max_len - 1] + "…"
            labels.append(f"{idx}: {text}")
        return labels

    roles_labels = make_labels(prompts["role"])
    instruct_labels = make_labels(prompts["instruct"])
    constraints_labels = make_labels(prompts["constraints"])
    return roles_labels, instruct_labels, constraints_labels


def save_fig(name: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"{name}.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_3d_role_instruct(df: pd.DataFrame, roles_labels: list[str], instruct_labels: list[str]) -> None:
    """3D: X=role, Y=instruct, Z=roc_auc; метки X/Y — части промптов."""
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")

    x = df["role"].to_numpy()
    y = df["instruct"].to_numpy()
    z = df["roc_auc"].to_numpy()

    norm = Normalize(vmin=z.min(), vmax=z.max())
    # colors = cm.viridis(norm(z))
    colors = cm.coolwarm(norm(z))

    ax.scatter(x, y, z, c=colors, s=30, alpha=0.8, depthshade=True)

    ax.set_xlabel("role")
    ax.set_ylabel("instruct")
    ax.set_zlabel("roc_auc")

    ax.set_xticks(range(len(roles_labels)))
    # ax.set_xticklabels(roles_labels, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(instruct_labels)))
    # ax.set_yticklabels(instruct_labels, rotation=45, ha="right", fontsize=7)

    ax.set_title("3D: role × instruct × roc_auc")

    # m = cm.ScalarMappable(cmap="viridis", norm=norm)
    m = cm.ScalarMappable(cmap="coolwarm", norm=norm)
    m.set_array([])
    fig.colorbar(m, ax=ax, shrink=0.6, pad=0.08, label="roc_auc")

    save_fig("20_3d_role_instruct_roc_auc")


def plot_3d_instruct_constraint(
    df: pd.DataFrame,
    instruct_labels: list[str],
    constraints_labels: list[str],
) -> None:
    """3D: X=instruct, Y=constraint, Z=roc_auc; метки X/Y — части промптов."""
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")

    x = df["instruct"].to_numpy()
    y = df["constraint"].to_numpy()
    z = df["roc_auc"].to_numpy()

    norm = Normalize(vmin=z.min(), vmax=z.max())
    # colors = cm.plasma(norm(z))
    colors = cm.coolwarm(norm(z))

    ax.scatter(x, y, z, c=colors, s=30, alpha=0.8, depthshade=True)

    ax.set_xlabel("instruct")
    ax.set_ylabel("constraint")
    ax.set_zlabel("roc_auc")

    ax.set_xticks(range(len(instruct_labels)))
    # ax.set_xticklabels(instruct_labels, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(constraints_labels)))
    # ax.set_yticklabels(constraints_labels, rotation=45, ha="right", fontsize=7)

    ax.set_title("3D: instruct × constraint × roc_auc")

    # m = cm.ScalarMappable(cmap="plasma", norm=norm)
    m = cm.ScalarMappable(cmap="coolwarm", norm=norm)
    m.set_array([])
    fig.colorbar(m, ax=ax, shrink=0.6, pad=0.08, label="roc_auc")

    save_fig("21_3d_instruct_constraint_roc_auc")


def plot_3d_role_constraint(
    df: pd.DataFrame,
    roles_labels: list[str],
    constraints_labels: list[str],
) -> None:
    """3D: X=role, Y=constraint, Z=roc_auc; метки X/Y — части промптов."""
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")

    x = df["role"].to_numpy()
    y = df["constraint"].to_numpy()
    z = df["roc_auc"].to_numpy()

    norm = Normalize(vmin=z.min(), vmax=z.max())
    colors = cm.coolwarm(norm(z))

    ax.scatter(x, y, z, c=colors, s=30, alpha=0.8, depthshade=True)

    ax.set_xlabel("role")
    ax.set_ylabel("constraint")
    ax.set_zlabel("roc_auc")

    ax.set_xticks(range(len(roles_labels)))
    # ax.set_xticklabels(roles_labels, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(constraints_labels)))
    # ax.set_yticklabels(constraints_labels, rotation=45, ha="right", fontsize=7)

    ax.set_title("3D: role × constraint × roc_auc")

    m = cm.ScalarMappable(cmap="coolwarm", norm=norm)
    m.set_array([])
    fig.colorbar(m, ax=ax, shrink=0.6, pad=0.08, label="roc_auc")

    save_fig("22_3d_role_constraint_roc_auc")





def plot_roc_auc_distribution(df: pd.DataFrame) -> None:
    """График распределения roc_auc: гистограмма + boxplot."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), 
                                    gridspec_kw={'height_ratios': [3, 1]})
    
    roc_auc_values = df["roc_auc"].to_numpy()
    
    n, bins, patches = ax1.hist(roc_auc_values, bins=30, alpha=0.7, 
                                 color='steelblue', edgecolor='black', linewidth=0.5)
    ax1.set_ylabel("Frequency", fontsize=11)
    ax1.set_title("Distribution of ROC-AUC Scores", fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    mean_val = np.mean(roc_auc_values)
    median_val = np.median(roc_auc_values)
    ax1.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.4f}')
    ax1.axvline(median_val, color='green', linestyle='--', linewidth=2, label=f'Median: {median_val:.4f}')
    ax1.legend(fontsize=10)
    
    stats_text = f'n = {len(roc_auc_values)}\n'
    stats_text += f'Mean = {mean_val:.4f}\n'
    stats_text += f'Std = {np.std(roc_auc_values):.4f}\n'
    stats_text += f'Min = {np.min(roc_auc_values):.4f}\n'
    stats_text += f'Max = {np.max(roc_auc_values):.4f}'
    ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes,
             fontsize=9, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    bp = ax2.boxplot(roc_auc_values, vert=False, patch_artist=True,
                     boxprops=dict(facecolor='lightblue', alpha=0.7),
                     medianprops=dict(color='red', linewidth=2))
    ax2.set_xlabel("ROC-AUC", fontsize=11)
    ax2.set_yticks([])
    ax2.grid(True, alpha=0.3, linestyle='--', axis='x')
    
    plt.suptitle("ROC-AUC Distribution Analysis", fontsize=16, fontweight='bold', y=0.98)
    
    save_fig("10_roc_auc_distribution")


def main() -> None:
    if not LOG_PATH.exists():
        raise FileNotFoundError(f"Нет файла {LOG_PATH}")
    if not PROMPTS_PATH.exists():
        raise FileNotFoundError(f"Нет файла {PROMPTS_PATH}")

    df = load_logs(LOG_PATH)
    if df.empty:
        raise ValueError(f"{LOG_PATH} пуст")

    roles_labels, instruct_labels, constraints_labels = load_prompt_labels(PROMPTS_PATH, max_len=35)

    plot_3d_role_instruct(df, roles_labels, instruct_labels)
    plot_3d_instruct_constraint(df, instruct_labels, constraints_labels)
    plot_3d_role_constraint(df, roles_labels, constraints_labels)

    plot_roc_auc_distribution(df)

    n_png = len(list(OUT_DIR.glob("*.png"))) if OUT_DIR.exists() else 0
    print(f"Сохранено файлов *.png в {OUT_DIR}/: {n_png}")


if __name__ == "__main__":
    main()

