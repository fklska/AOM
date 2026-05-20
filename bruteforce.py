import pandas as pd
import random
import json
from agent import build_prompt, execute
import tqdm

import warnings
warnings.filterwarnings("ignore")

random.seed(42)
prompts = pd.read_csv("prompts.csv", sep=";")
roles = prompts.role.to_list()
instructs = prompts.instruct.to_list()
constraints = prompts.constraints.to_list()

result = []
for role_idx in tqdm.tqdm(range(8, 9)):
    for task_idx in range(2, 3):
        for constraint_idx in range(7, 8):
            prompt = build_prompt(roles[role_idx], instructs[task_idx], constraints[constraint_idx])
            auc = execute(prompt)
            print(auc)
            log_entry = {
                "role": role_idx,
                "instruct": task_idx,
                "constraint": constraint_idx,
                "roc_auc": auc
            }
            result.append(log_entry)
            with open("brute_force.json", "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")