import json
from pathlib import Path
import pandas as pd
from agent import build_prompt, execute

LOG_PATH = Path("logs_bf.json")
PROMPTS_PATH = "prompts.csv"

START_ROLE: int | None = None
START_INSTRUCT: int | None = None
START_CONSTRAINT: int | None = None


def load_prompt_parts():
    prompts = pd.read_csv(PROMPTS_PATH, sep=";")
    return (
        prompts.role.to_list(),
        prompts.instruct.to_list(),
        prompts.constraints.to_list(),
    )


def load_completed_triples(log_path: Path) -> set[tuple[int, int, int]]:
    if not log_path.exists():
        return set()
    completed = set()
    with log_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            completed.add((entry["role"], entry["instruct"], entry["constraint"]))
    return completed


def append_log(entry: dict) -> None:
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def iter_triples(
    start_role: int = 0,
    start_instruct: int = 0,
    start_constraint: int = 0,
):
    for role_idx in range(start_role, 10):
        instruct_from = start_instruct if role_idx == start_role else 0
        for instruct_idx in range(instruct_from, 10):
            constraint_from = (
                start_constraint
                if role_idx == start_role and instruct_idx == start_instruct
                else 0
            )
            for constraint_idx in range(constraint_from, 10):
                yield role_idx, instruct_idx, constraint_idx


def main():
    roles, instructs, constraints = load_prompt_parts()
    completed = load_completed_triples(LOG_PATH)

    start_role = 0 if START_ROLE is None else START_ROLE
    start_instruct = 0 if START_INSTRUCT is None else START_INSTRUCT
    start_constraint = 0 if START_CONSTRAINT is None else START_CONSTRAINT

    for role_idx, instruct_idx, constraint_idx in iter_triples(
        start_role, start_instruct, start_constraint
        # start_role = 9, start_instruct = 9, start_constraint = 8

    ):
        triple = (role_idx, instruct_idx, constraint_idx)
        if triple in completed:
            print(f"skip {role_idx} {instruct_idx} {constraint_idx}")
            continue

        print(role_idx, instruct_idx, constraint_idx)
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
        completed.add(triple)


if __name__ == "__main__":
    main()
