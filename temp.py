from pathlib import Path
from agent import build_prompt, execute
from bruteforce import append_log, load_completed_triples, load_prompt_parts

LOG_PATH = Path("logs_bf copy.json")

roles, instructs, constraints = load_prompt_parts()
# completed = load_completed_triples(LOG_PATH)

role_idx = 8
instruct_idx = 2
constraint_idx = 7
triple = (role_idx, instruct_idx, constraint_idx)

prompt = build_prompt(
        roles[role_idx],
        instructs[instruct_idx],
        constraints[constraint_idx],
        )
auc = execute(prompt)
print(auc)
print("=" * 10)

log_entry = {
    "role": role_idx,
    "instruct": instruct_idx,
    "constraint": constraint_idx,
    "roc_auc": auc,
}
append_log(log_entry)
# completed.add(triple)