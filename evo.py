import pandas as pd
import random
import numpy as np
from deap import base, creator, tools, algorithms
import json
from agent import build_prompt, execute

prompts = pd.read_csv("prompts.csv", sep=";")
roles = prompts.role.to_list()
instructs = prompts.instruct.to_list()
constraints = prompts.constraints.to_list()


creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)


toolbox = base.Toolbox()

toolbox.register("attr_role", random.randint, 0, 9)
toolbox.register("attr_task", random.randint, 0, 9)
toolbox.register("attr_constraint", random.randint, 0, 9)

toolbox.register("individual", tools.initCycle, creator.Individual, (toolbox.attr_role, toolbox.attr_task, toolbox.attr_constraint), n=1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)


def fitness(individual):
    role_idx, task_idx, constraint_idx = individual
    prompt = build_prompt(roles[role_idx], instructs[task_idx], constraints[constraint_idx])
    print(role_idx, task_idx, constraint_idx)
    auc = execute(prompt)
    print(auc)
    print("=" * 10)
    log_entry = {
        "individual": individual,
        "role": role_idx,
        "instruct": task_idx,
        "constraint": constraint_idx,
        "roc_auc": auc
    }

    with open("logs.json", "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

    return (auc,)

toolbox.register("evaluate", fitness)

toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutUniformInt, low=0, up=9, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)


def main():
    random.seed(42)
    pop = toolbox.population(n=10)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("avg", np.mean)
    stats.register("max", np.max)
    stats.register("min", np.min)

    pop, logbook = algorithms.eaSimple(pop, toolbox, cxpb=0.7, mutpb=0.3, ngen=3, stats=stats, halloffame=hof, verbose=True)
    return pop, hof, logbook



if __name__ == "__main__":
    pop, hof, logbook = main()
    best = hof[0]
    best_prompt = f"{roles[best[0]]}\n\n{instructs[best[1]]}\n\n{constraints[best[2]]}"
    print("\nЛучшая особь (индексы):", best)
    print("Лучший ROC-AUC:", best.fitness.values[0])
    print("Лучший промпт:\n", best_prompt)