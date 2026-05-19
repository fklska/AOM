import json
from pathlib import Path

LOG_PATH = Path("logs_bf.json")

entries = [
    json.loads(line)
    for line in LOG_PATH.read_text(encoding="utf-8").splitlines()
    if line.strip()
]

top3 = sorted(entries, key=lambda e: e["roc_auc"], reverse=True)[:3]

for i, entry in enumerate(top3, start=1):
    print(f"{i}. roc_auc={entry['roc_auc']}")
    print(json.dumps(entry, ensure_ascii=False))
