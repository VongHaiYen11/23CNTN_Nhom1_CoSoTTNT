import numpy as np
import time
import pandas as pd

from src.algorithms.swarm_algorithms.ABC import ArtificialBeeColonyKnapsack
from src.algorithms.swarm_algorithms.FA import FireflyKnapsack
from src.algorithms.swarm_algorithms.PSO import ParitcleSwarmKnapsack
from src.algorithms.swarm_algorithms.Cuckoo import CuckooSearchKnapsack
from src.algorithms.swarm_algorithms.ACO import AntColonyOptimizationKnapsack
from src.algorithms.traditional_algorithms.GA import GeneticAlgorithmKnapsack
from src.algorithms.traditional_algorithms.HC import HillClimbingKnapsack
from src.algorithms.traditional_algorithms.SA import SimulatedAnnealingKnapsack
from src.problem.discrete.knapsack import WEIGHTS, VALUES, MAX_WEIGHT, N_ITEMS

POP_SIZE = 10
MAX_ITER = 200
SEED = 42
VERBOSE = False

results = {}


def compute_solution_metrics(solution):
    binary = (solution > 0).astype(int) if solution.dtype != int else solution
    return {
        'value': np.sum(binary * VALUES),
        'weight': np.sum(binary * WEIGHTS),
        'binary': binary
    }


def run_algorithm_and_store(name, algorithm, print_name=None):
    if print_name:
        print(f"\n--- {print_name} ---")
    
    start = time.time()
    solution, fitness, history = algorithm.run()
    elapsed = time.time() - start
    
    metrics = compute_solution_metrics(solution)
    results[name] = {
        'value': metrics['value'],
        'weight': metrics['weight'],
        'binary': metrics['binary'],
        'time': elapsed
    }


def print_results(results_dict):
    rows = []
    for algo_name, res in results_dict.items():
        rows.append({
            'Algorithm': algo_name,
            'Value': res['value'],
            'Weight': res['weight'],
            'Time (s)': res.get('time', 0),
            'Solution': ''.join(map(str, res['binary']))
        })
    
    df = pd.DataFrame(rows)
    df = df.sort_values(by='Value', ascending=False).reset_index(drop=True)
    
    df['Value'] = df['Value'].apply(lambda x: f"{x:.2f}")
    df['Weight'] = df['Weight'].apply(lambda x: f"{x:.2f}")
    df['Time (s)'] = df['Time (s)'].apply(lambda x: f"{x:.4f}")
    
    print("\n" + "=" * 100)
    print(" " * 32 + "Knapsack Optimization Results")
    print("=" * 100)
    
    col_widths = {
        'Algorithm': 15,
        'Value': 12,
        'Weight': 12,
        'Time (s)': 12,
        'Solution': 30
    }
    
    header = f"| {'Algorithm':<{col_widths['Algorithm']}} | {'Value':<{col_widths['Value']}} | {'Weight':<{col_widths['Weight']}} | {'Time (s)':<{col_widths['Time (s)']}} | {'Solution':<{col_widths['Solution']}} |"
    separator = "|" + "-" * (col_widths['Algorithm'] + 2) + "|" + "-" * (col_widths['Value'] + 2) + "|" + "-" * (col_widths['Weight'] + 2) + "|" + "-" * (col_widths['Time (s)'] + 2) + "|" + "-" * (col_widths['Solution'] + 2) + "|"
    
    print(header)
    print(separator)
    
    for _, row in df.iterrows():
        solution_str = str(row['Solution'])
        if len(solution_str) > col_widths['Solution']:
            solution_str = solution_str[:col_widths['Solution']-3] + "..."
        line = f"| {str(row['Algorithm']):<{col_widths['Algorithm']}} | {str(row['Value']):<{col_widths['Value']}} | {str(row['Weight']):<{col_widths['Weight']}} | {str(row['Time (s)']):<{col_widths['Time (s)']}} | {solution_str:<{col_widths['Solution']}} |"
        print(line)
    
    print("=" * 100)
    print()


def run_knapsack():
    print("\n--- Running Experiments - Knapsack Problem ---")
    print(f"Pop Size = {POP_SIZE}, Max Iterations = {MAX_ITER}, Items = {N_ITEMS}\n")
    
    run_algorithm_and_store(
        'ABC',
        ArtificialBeeColonyKnapsack(
            weights=WEIGHTS,
            values=VALUES,
            max_weight=MAX_WEIGHT,
            dim=N_ITEMS,
            num_employed_bees=POP_SIZE,
            num_onlooker_bees=POP_SIZE,
            max_iter=MAX_ITER * 5,
            limit=30,
            seed=SEED,
            verbose=VERBOSE
        )
    )
    
    run_algorithm_and_store(
        'FA',
        FireflyKnapsack(
            weights=WEIGHTS,
            values=VALUES,
            max_weight=MAX_WEIGHT,
            population_size=POP_SIZE,
            max_iter=MAX_ITER * 5,
            seed=SEED,
            verbose=VERBOSE
        )
    )
    
    run_algorithm_and_store(
        'CS',
        CuckooSearchKnapsack(
            weights=WEIGHTS,
            values=VALUES,
            capacity=MAX_WEIGHT,
            dim=N_ITEMS,
            population_size=POP_SIZE,
            max_iter=MAX_ITER * 5,
            seed=SEED,
            verbose=VERBOSE
        ),
        print_name="Cuckoo Search"
    )
    
    run_algorithm_and_store(
        'PSO',
        ParitcleSwarmKnapsack(
            weights=WEIGHTS,
            values=VALUES,
            capacity=MAX_WEIGHT,
            dim=N_ITEMS,
            population_size=POP_SIZE,
            max_iter=MAX_ITER,
            seed=SEED,
            verbose=VERBOSE
        )
    )
    
    run_algorithm_and_store(
        'ACO',
        AntColonyOptimizationKnapsack(
            weights=WEIGHTS,
            values=VALUES,
            capacity=MAX_WEIGHT,
            n_ants=POP_SIZE,
            max_iter=MAX_ITER,
            seed=SEED,
            verbose=VERBOSE
        )
    )
    
    run_algorithm_and_store(
        'HC',
        HillClimbingKnapsack(
            weights=WEIGHTS,
            values=VALUES,
            capacity=MAX_WEIGHT,
            dim=N_ITEMS,
            max_iter=MAX_ITER,
            n_neighbors=POP_SIZE,
            seed=SEED,
            verbose=VERBOSE
        ),
        print_name="Hill Climbing"
    )
    
    run_algorithm_and_store(
        'SA',
        SimulatedAnnealingKnapsack(
            weights=WEIGHTS,
            values=VALUES,
            capacity=MAX_WEIGHT,
            dim=N_ITEMS,
            max_iter=MAX_ITER * 2,
            step_size=0.5,
            initial_temp=100,
            seed=SEED,
            verbose=VERBOSE
        ),
        print_name="Simulated Annealing"
    )
    
    run_algorithm_and_store(
        'GA',
        GeneticAlgorithmKnapsack(
            weights=WEIGHTS,
            values=VALUES,
            capacity=MAX_WEIGHT,
            population_size=POP_SIZE,
            max_iter=MAX_ITER,
            seed=SEED,
            verbose=VERBOSE
        ),
        print_name="Genetic Algorithm"
    )


if __name__ == "__main__":
    run_knapsack()
    print_results(results)
