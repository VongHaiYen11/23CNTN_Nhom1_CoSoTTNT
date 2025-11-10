import numpy as np
import time
import os
import pandas as pd

# ==== Import các thuật toán ====

from src.algorithms.swarm_algorithms.ABC import ArtificialBeeColony
from src.algorithms.swarm_algorithms.FA import  FireflyAlgorithm
from src.algorithms.swarm_algorithms.PSO import ParticleSwarmOptimization
from src.algorithms.swarm_algorithms.Cuckoo import CuckooSearch
from src.algorithms.swarm_algorithms.ACO import AntColonyOptimizationKnapsack
from src.algorithms.traditional_algorithms.GA import GeneticAlgorithmContinuos
from src.algorithms.traditional_algorithms.HC import HillClimbing
from src.algorithms.traditional_algorithms.SA import SimulatedAnnealing

# ==== Import bài toán ====
from src.problem.discrete.knapsack import WEIGHTS, VALUES, MAX_WEIGHT, N_ITEMS
from src.experiment.knapsack_adapter import knapsack_fitness_adapter

# Cấu hình chung
POP_SIZE = 100
MAX_ITER = 200
SEED = 42
LB, UB = -10, 10

results = {}

def knapsack_fitness_continuos(x):
    return knapsack_fitness_adapter(x, WEIGHTS, VALUES, MAX_WEIGHT, use_sigmoid=False, seed=SEED)

def knapsack_fitness_discrete(x):
    return knapsack_fitness_adapter(x, WEIGHTS, VALUES, MAX_WEIGHT, use_sigmoid=False, seed=SEED)

def print_results(results_dict):
    """
    In kết quả cuối cùng của các thuật toán cho Knapsack problem
    """
    rows = []
    for algo_name, res in results_dict.items():
        row = {
            'Algorithm': algo_name,
            'Value': res['value'],
            'Weight': res['weight'],
            'Solution': ''.join(map(str, res['binary']))  # hiển thị nghiệm 0/1
        }
        rows.append(row)
    
    df = pd.DataFrame(rows)
    df = df.sort_values(by='Value', ascending=False).reset_index(drop=True)
    
    print("\n=== Knapsack Optimization Results ===")
    print(df.to_string(index=False))


def run_knapsack():
    print("\n--- CHẠY THÍ NGHIỆM - KNAPSACK PROBLEM ---")
    print(f"Pop Size = {POP_SIZE}, Max Iterations = {MAX_ITER}, Items = {N_ITEMS}\n")

    # === ABC ===
    abc = ArtificialBeeColony(
        fitness_function=knapsack_fitness_continuos,
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
        objective_function=knapsack_fitness_continuos,
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
        fitness_func=knapsack_fitness_continuos,
        lower_bound=LB, upper_bound=UB,
        dim=N_ITEMS,
        population_size=POP_SIZE,
        max_iter=MAX_ITER,
        seed=SEED,
        verbose=True
    )
    sol_cs, fit_cs = cs.run()
    print(sol_cs)
    bin_cs = (sol_cs > 0).astype(int)
    results['CS'] = {'value': np.sum(bin_cs * VALUES), 'weight': np.sum(bin_cs * WEIGHTS), 'binary': bin_cs}

    # === PSO ===
    pso = ParticleSwarmOptimization(
        objective_function=knapsack_fitness_continuos,
        lower_bound=LB, upper_bound=UB,
        dim=N_ITEMS,
        population_size=POP_SIZE,
        max_iter=MAX_ITER,
        seed=SEED
    )
    sol_pso, fit_pso = pso.run()
    bin_pso = (sol_pso > 0).astype(int)
    val_pso = np.sum(bin_pso * VALUES)
    w_pso = np.sum(bin_pso * WEIGHTS)
    results['PSO'] = {'value': val_pso, 'weight': w_pso, 'binary': bin_pso}

    # === ACO ===
    aco = AntColonyOptimizationKnapsack (
        fitness_function=knapsack_fitness_discrete,
        weights=WEIGHTS,
        values=VALUES,
        capacity=MAX_WEIGHT,
        n_ants=POP_SIZE,
        max_iter=MAX_ITER,
        alpha=1,
        beta=2,
        rho=0.3,
        Q=1,
        seed=SEED
    )
    sol_aco, fit_aco = aco.run()
    bin_aco = (sol_aco > 0).astype(int)
    val_aco = np.sum(bin_aco * VALUES)
    w_aco = np.sum(bin_aco * WEIGHTS)
    results['ACO'] = {'value': val_aco, 'weight': w_aco, 'binary': bin_aco}

    # === Hill Climbing ===
    print("--- Hill Climbing ---")
    hc = HillClimbing(
        fitness_func=knapsack_fitness_continuos,
        lower_bound=LB,
        upper_bound=UB,
        dim=N_ITEMS,
        max_iter=MAX_ITER,
        n_neighbors=POP_SIZE,  # số neighbors = pop size
        tolerance=1e-4,
        seed=SEED,
        verbose=False
    )
    sol_hc, fit_hc = hc.run()
    bin_hc = (sol_hc > 0).astype(int)
    val_hc = np.sum(bin_hc * VALUES)
    w_hc = np.sum(bin_hc * WEIGHTS)
    results['HC'] = {'value': val_hc, 'weight': w_hc, 'binary': bin_hc}

    # === Simulated Annealing ===
    print("--- Simulated Annealing ---")
    sa = SimulatedAnnealing(
        fitness_func=knapsack_fitness_continuos,
        lower_bound=LB,
        upper_bound=UB,
        dim=N_ITEMS,
        max_iter=MAX_ITER * 2,  # SA thường cần nhiều iter hơn
        step_size=0.5,
        initial_temp=100,
        seed=SEED,
        verbose=True
    )
    sol_sa, fit_sa = sa.run()
    bin_sa = (sol_sa > 0).astype(int)
    val_sa = np.sum(bin_sa * VALUES)
    w_sa = np.sum(bin_sa * WEIGHTS)
    results['SA'] = {'value': val_sa, 'weight': w_sa, 'binary': bin_sa}


if __name__ == "__main__":
    run_knapsack()
    print_results(results)
