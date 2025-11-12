import numpy as np
import time
import os
import pandas as pd

# ==== Import các thuật toán ====

from src.algorithms.swarm_algorithms.ABC import ArtificialBeeColony
from src.algorithms.swarm_algorithms.FA import  FireflyAlgorithm
from src.algorithms.swarm_algorithms.PSO import ParticleSwarmOptimization
from src.algorithms.swarm_algorithms.Cuckoo import CuckooSearchKnapsack
from src.algorithms.swarm_algorithms.ACO import AntColonyOptimizationKnapsack
from src.algorithms.traditional_algorithms.GA import GeneticAlgorithmKnapsack
from src.algorithms.traditional_algorithms.HC import HillClimbingKnapsack
from src.algorithms.traditional_algorithms.SA import SimulatedAnnealingKnapsack

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
    In kết quả cuối cùng của các thuật toán cho Knapsack problem, kèm thời gian chạy
    """
    rows = []
    for algo_name, res in results_dict.items():
        row = {
            'Algorithm': algo_name,
            'Value': res['value'],
            'Weight': res['weight'],
            'Time (s)': round(res.get('time', 0), 4),  # lấy thời gian nếu có, làm tròn 4 chữ số
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
    start = time.time()
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
    end = time.time()
    bin_abc = (sol_abc > 0).astype(int)
    results['ABC'] = {
        'value': np.sum(bin_abc * VALUES),
        'weight': np.sum(bin_abc * WEIGHTS),
        'binary': bin_abc,
        'time': end - start
    }

    # === FA ===
    start = time.time()
    fa = FireflyAlgorithm(
        objective_function=knapsack_fitness_continuos,
        lower_bound=LB, upper_bound=UB,
        dimension=N_ITEMS,
        population_size=POP_SIZE,
        max_iterations=MAX_ITER,
        seed=SEED
    )
    sol_fa, fit_fa, hist_fa = fa.run()
    end = time.time()
    bin_fa = (sol_fa > 0).astype(int)
    results['FA'] = {
        'value': np.sum(bin_fa * VALUES),
        'weight': np.sum(bin_fa * WEIGHTS),
        'binary': bin_fa,
        'time': end - start
    }

    # === Cuckoo Search ===
    print("\n--- Cuckoo Search ---")
    start = time.time()
    cs = CuckooSearchKnapsack(
        weights=WEIGHTS,
        values=VALUES,
        capacity=MAX_WEIGHT,
        dim=N_ITEMS,
        population_size=POP_SIZE,
        max_iter=MAX_ITER * 5,
        seed=SEED,
        verbose=True
    )
    sol_cs, fit_cs = cs.run()
    print(sol_cs)
    end = time.time()
    results['CS'] = {
        'value': np.sum(sol_cs * VALUES),
        'weight': np.sum(sol_cs * WEIGHTS),
        'binary': sol_cs,
        'time': end - start
    }

    # === PSO ===
    start = time.time()
    pso = ParticleSwarmOptimization(
        objective_function=knapsack_fitness_continuos,
        lower_bound=LB, upper_bound=UB,
        dim=N_ITEMS,
        population_size=POP_SIZE,
        max_iter=MAX_ITER,
        seed=SEED
    )
    sol_pso, fit_pso = pso.run()
    end = time.time()
    bin_pso = (sol_pso > 0).astype(int)
    results['PSO'] = {
        'value': np.sum(bin_pso * VALUES),
        'weight': np.sum(bin_pso * WEIGHTS),
        'binary': bin_pso,
        'time': end - start
    }

    # === ACO ===
    start = time.time()
    aco = AntColonyOptimizationKnapsack(
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
    end = time.time()
    bin_aco = (sol_aco > 0).astype(int)
    results['ACO'] = {
        'value': np.sum(bin_aco * VALUES),
        'weight': np.sum(bin_aco * WEIGHTS),
        'binary': bin_aco,
        'time': end - start
    }

    # === Hill Climbing ===
    print("--- Hill Climbing ---")
    start = time.time()
    hc = HillClimbingKnapsack(
        weights=WEIGHTS,
        values=VALUES,
        capacity=MAX_WEIGHT,
        dim=N_ITEMS,
        max_iter=MAX_ITER,
        n_neighbors=POP_SIZE,
        seed=SEED,
        verbose=False
    )
    sol_hc, fit_hc = hc.run()
    end = time.time()
    results['HC'] = {
        'value': np.sum(sol_hc * VALUES),
        'weight': np.sum(sol_hc * WEIGHTS),
        'binary': sol_hc,
        'time': end - start
    }

    # === Simulated Annealing ===
    print("--- Simulated Annealing ---")
    start = time.time()
    sa = SimulatedAnnealingKnapsack(
        weights=WEIGHTS,
        values=VALUES,
        capacity=MAX_WEIGHT,
        dim=N_ITEMS,
        max_iter=MAX_ITER * 2,
        step_size=0.5,
        initial_temp=100,
        seed=SEED,
        verbose=True
    )
    sol_sa, fit_sa = sa.run()
    end = time.time()
    results['SA'] = {
        'value': np.sum(sol_sa * VALUES),
        'weight': np.sum(sol_sa * WEIGHTS),
        'binary': sol_sa,
        'time': end - start
    }

if __name__ == "__main__":
    run_knapsack()
    print_results(results)
