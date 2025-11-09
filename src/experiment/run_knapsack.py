import numpy as np
import time
import os
import pandas as pd

# ==== Import các thuật toán ====

from src.algorithms.swarm_algorithms.ABC import ArtificialBeeColony
from src.algorithms.swarm_algorithms.FA import  FireflyAlgorithm
from src.algorithms.swarm_algorithms.PSO import pso_optimize
from src.algorithms.swarm_algorithms.Cuckoo import CuckooSearch

# ==== Import bài toán ====
from src.problem.discrete.knapsack import WEIGHTS, VALUES, MAX_WEIGHT, N_ITEMS
from src.experiment.knapsack_adapter import knapsack_fitness_adapter

def knapsack_fitness(x):
    return knapsack_fitness_adapter(x, WEIGHTS, VALUES, MAX_WEIGHT, use_sigmoid=True)

# Cấu hình chung
POP_SIZE = 100
MAX_ITER = 200
SEED = 42
LB, UB = -10, 10

results = {}


def run_knapsack():
    print("\n--- CHẠY THÍ NGHIỆM - KNAPSACK PROBLEM ---")
    print(f"Pop Size = {POP_SIZE}, Max Iterations = {MAX_ITER}, Items = {N_ITEMS}\n")

    # === ABC ===
    abc = ArtificialBeeColony(
        fitness_function=knapsack_fitness,
        lower_bound=LB, upper_bound=UB,
        problem_size=N_ITEMS,
        num_employed_bees=POP_SIZE//2,
        num_onlooker_bees=POP_SIZE//2,
        max_iterations=MAX_ITER,
        limit=50,
        seed=SEED
    )
    sol_abc, fit_abc, hist_abc = abc.run()
    bin_abc = (sol_abc > 0).astype(int)
    val_abc = np.sum(bin_abc * VALUES)
    w_abc = np.sum(bin_abc * WEIGHTS)
    results['ABC'] = {'value': val_abc, 'weight': w_abc, 'binary': bin_abc}

    # === FA ===
    fa = FireflyAlgorithm(
        objective_function=knapsack_fitness,
        lower_bound=LB, upper_bound=UB,
        dimension=N_ITEMS,
        population_size=POP_SIZE,
        max_iterations=MAX_ITER,
        seed=SEED
    )
    sol_fa, fit_fa, hist_fa = fa.run()
    bin_fa = (sol_fa > 0).astype(int)
    val_fa = np.sum(bin_fa * VALUES)
    w_fa = np.sum(bin_fa * WEIGHTS)
    results['FA'] = {'value': val_fa, 'weight': w_fa, 'binary': bin_fa}

    # === Cuckoo Search ===
    cs = CuckooSearch(
        fitness_func=knapsack_fitness,
        lower_bound=LB, upper_bound=UB,
        dim=N_ITEMS,
        population_size=POP_SIZE,
        max_iter=MAX_ITER,
        seed=SEED
    )
    sol_cs, fit_cs = cs.run()
    bin_cs = (sol_cs > 0).astype(int)
    results['CS'] = {'value': np.sum(bin_cs * VALUES), 'weight': np.sum(bin_cs * WEIGHTS)}

if __name__ == "__main__":
    run_knapsack()
