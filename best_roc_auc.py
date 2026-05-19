import json
from pathlib import Path

best_line = None
best_auc = float("-inf")

with Path("logs_bf.json").open(encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        entry = json.loads(line)
        auc = entry["roc_auc"]
        if auc > best_auc:
            best_auc = auc
            best_line = line

print("Лучший roc_auc:", best_auc)
print("Строка:", best_line)