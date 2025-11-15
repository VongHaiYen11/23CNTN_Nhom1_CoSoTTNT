import numpy as np
import time
import os
import pandas as pd
import sys
import random

from src.algorithms.swarm_algorithms.ABC import ArtificialBeeColonyKnapsack
from src.algorithms.swarm_algorithms.FA import FireflyKnapsack
from src.algorithms.swarm_algorithms.PSO import ParitcleSwarmKnapsack
from src.algorithms.swarm_algorithms.Cuckoo import CuckooSearchKnapsack
from src.algorithms.swarm_algorithms.ACO import AntColonyOptimizationKnapsack
from src.algorithms.traditional_algorithms.GA import GeneticAlgorithmKnapsack
from src.algorithms.traditional_algorithms.HC import HillClimbingKnapsack
from src.algorithms.traditional_algorithms.SA import SimulatedAnnealingKnapsack
from src.problem.discrete.knapsack import WEIGHTS, VALUES, MAX_WEIGHT, N_ITEMS

N_RUNS = 1
POP_SIZES = [10]
MAX_ITERATIONS = 200
SEED = 42
VERBOSE = False

ALGOS = ['ABC', 'FA', 'Cuckoo', 'PSO', 'ACO', 'HC', 'SA', 'GA']


def measure_space_usage(obj):
    if isinstance(obj, np.ndarray):
        return obj.nbytes + sys.getsizeof(obj)
    elif isinstance(obj, list):
        return sys.getsizeof(obj) + sum(measure_space_usage(o) for o in obj)
    return sys.getsizeof(obj)


def create_algorithm(algo_name, pop_size, seed):
    if algo_name == 'ABC':
        return ArtificialBeeColonyKnapsack(
            weights=WEIGHTS,
            values=VALUES,
            max_weight=MAX_WEIGHT,
            dim=N_ITEMS,
            num_employed_bees=pop_size,
            num_onlooker_bees=pop_size,
            max_iter=MAX_ITERATIONS,
            limit=30,
            seed=seed,
            verbose=VERBOSE
        )
    elif algo_name == 'FA':
        return FireflyKnapsack(
            weights=WEIGHTS,
            values=VALUES,
            max_weight=MAX_WEIGHT,
            population_size=pop_size,
            max_iter=MAX_ITERATIONS,
            seed=seed,
            verbose=VERBOSE
        )
    elif algo_name == 'Cuckoo':
        return CuckooSearchKnapsack(
            weights=WEIGHTS,
            values=VALUES,
            capacity=MAX_WEIGHT,
            dim=N_ITEMS,
            population_size=pop_size,
            max_iter=MAX_ITERATIONS,
            seed=seed,
            verbose=VERBOSE
        )
    elif algo_name == 'PSO':
        return ParitcleSwarmKnapsack(
            weights=WEIGHTS,
            values=VALUES,
            capacity=MAX_WEIGHT,
            dim=N_ITEMS,
            population_size=pop_size,
            max_iter=MAX_ITERATIONS,
            seed=seed,
            verbose=VERBOSE
        )
    elif algo_name == 'ACO':
        return AntColonyOptimizationKnapsack(
            weights=WEIGHTS,
            values=VALUES,
            capacity=MAX_WEIGHT,
            n_ants=pop_size,
            max_iter=MAX_ITERATIONS,
            seed=seed,
            verbose=VERBOSE
        )
    elif algo_name == 'HC':
        return HillClimbingKnapsack(
            weights=WEIGHTS,
            values=VALUES,
            capacity=MAX_WEIGHT,
            dim=N_ITEMS,
            max_iter=MAX_ITERATIONS,
            n_neighbors=pop_size,
            seed=seed,
            verbose=VERBOSE
        )
    elif algo_name == 'SA':
        return SimulatedAnnealingKnapsack(
            weights=WEIGHTS,
            values=VALUES,
            capacity=MAX_WEIGHT,
            dim=N_ITEMS,
            max_iter=MAX_ITERATIONS,
            step_size=0.5,
            initial_temp=100,
            seed=seed,
            verbose=VERBOSE
        )
    elif algo_name == 'GA':
        return GeneticAlgorithmKnapsack(
            weights=WEIGHTS,
            values=VALUES,
            capacity=MAX_WEIGHT,
            population_size=pop_size,
            max_iter=MAX_ITERATIONS,
            seed=seed,
            verbose=VERBOSE
        )
    else:
        raise ValueError(f"Unknown algorithm: {algo_name}")


def compute_solution_metrics(solution):
    binary = (solution > 0).astype(int) if solution.dtype != int else solution
    return {
        'value': np.sum(binary * VALUES),
        'weight': np.sum(binary * WEIGHTS),
        'binary': binary
    }


def run_algorithm(algo_name, pop_size, seed):
    np.random.seed(seed)
    random.seed(seed)
    start_time = time.time()
    
    algorithm = create_algorithm(algo_name, pop_size, seed)
    result = algorithm.run()
    
    solution, fitness, hist = result
    metrics = compute_solution_metrics(solution)
    
    elapsed = time.time() - start_time
    space = measure_space_usage(solution)
    
    return {
        'value': metrics['value'],
        'weight': metrics['weight'],
        'solution': ''.join(map(str, metrics['binary'])),
        'elapsed': elapsed,
        'space': space
    }


def print_results(results_list):
    df = pd.DataFrame(results_list)
    df = df.sort_values(by='Mean Value', ascending=False).reset_index(drop=True)
    
    df['Mean Value'] = df['Mean Value'].apply(lambda x: f"{x:.2f}")
    df['Mean Weight'] = df['Mean Weight'].apply(lambda x: f"{x:.2f}")
    df['Mean Time (s)'] = df['Mean Time (s)'].apply(lambda x: f"{x:.4f}")
    df['Mean Space (bytes)'] = df['Mean Space (bytes)'].apply(lambda x: f"{x:,.0f}")
    
    print("\n" + "=" * 100)
    print(" " * 32 + "Knapsack Optimization Results")
    print("=" * 100)
    
    col_widths = {
        'Algorithm': 15,
        'Mean Value': 12,
        'Mean Weight': 12,
        'Mean Time (s)': 12,
        'Mean Space (bytes)': 20,
        'Solution': 30
    }
    
    header = f"| {'Algorithm':<{col_widths['Algorithm']}} | {'Mean Value':<{col_widths['Mean Value']}} | {'Mean Weight':<{col_widths['Mean Weight']}} | {'Mean Time (s)':<{col_widths['Mean Time (s)']}} | {'Mean Space (bytes)':<{col_widths['Mean Space (bytes)']}} | {'Solution':<{col_widths['Solution']}} |"
    separator = "|" + "-" * (col_widths['Algorithm'] + 2) + "|" + "-" * (col_widths['Mean Value'] + 2) + "|" + "-" * (col_widths['Mean Weight'] + 2) + "|" + "-" * (col_widths['Mean Time (s)'] + 2) + "|" + "-" * (col_widths['Mean Space (bytes)'] + 2) + "|" + "-" * (col_widths['Solution'] + 2) + "|"
    
    print(header)
    print(separator)
    
    for _, row in df.iterrows():
        solution_str = str(row['Solution'])
        if len(solution_str) > col_widths['Solution']:
            solution_str = solution_str[:col_widths['Solution']-3] + "..."
        line = f"| {str(row['Algorithm']):<{col_widths['Algorithm']}} | {str(row['Mean Value']):<{col_widths['Mean Value']}} | {str(row['Mean Weight']):<{col_widths['Mean Weight']}} | {str(row['Mean Time (s)']):<{col_widths['Mean Time (s)']}} | {str(row['Mean Space (bytes)']):<{col_widths['Mean Space (bytes)']}} | {solution_str:<{col_widths['Solution']}} |"
        print(line)
    
    print("=" * 100)
    print()


def run_experiments():
    os.makedirs("results", exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    
    results = []
    print("\n--- Running Experiments ---")
    
    for algo in ALGOS:
        values, weights, solutions, times, spaces = [], [], [], [], []
        
        for pop in POP_SIZES:
            for _ in range(N_RUNS):
                seed = random.randint(0, SEED)
                res = run_algorithm(algo, pop, seed)
                values.append(res['value'])
                weights.append(res['weight'])
                solutions.append(res['solution'])
                times.append(res['elapsed'])
                spaces.append(res['space'])
        
        results.append({
            "Algorithm": algo,
            'Mean Value': np.mean(values),
            'Mean Weight': np.mean(weights),
            'Solution': solutions[0] if solutions else '',
            'Mean Time (s)': np.mean(times),
            'Mean Space (bytes)': np.mean(spaces)
        })
    
    print_results(results)
    pd.DataFrame(results).to_csv(f"results/resultsKnapsack_{timestamp}.csv", index=False)
    print("Result CSV saved!")
    print("\nFinish")


if __name__ == "__main__":
    run_experiments()
