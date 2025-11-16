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
from src.problem.discrete.knapsack import WEIGHTS, VALUES, MAX_WEIGHT, N_ITEMS, TEST_CASES

N_RUNS = 1
POP_SIZES = [50]
MAX_ITERATIONS = 200
SEED = 42
VERBOSE = False

ALGOS = ['ABC', 'FA', 'Cuckoo', 'PSO', 'ACO', 'HC', 'SA', 'GA']


def measure_space_usage(obj):
    """Calculate memory usage of an object recursively.

    Parameters:
    obj: Object to measure (numpy array, list, or other)

    Returns:
    int: Total memory usage in bytes
    """
    if isinstance(obj, np.ndarray):
        return obj.nbytes + sys.getsizeof(obj)
    elif isinstance(obj, list):
        return sys.getsizeof(obj) + sum(measure_space_usage(o) for o in obj)
    return sys.getsizeof(obj)


def create_algorithm(algo_name, pop_size, seed, weights=None, values=None, max_weight=None, n_items=None):
    """Create algorithm instance for knapsack problem optimization.

    Parameters:
    algo_name (str): Name of the algorithm (ABC, FA, Cuckoo, PSO, ACO, HC, SA, GA)
    pop_size (int): Population size
    seed (int): Random seed for reproducibility
    weights (np.ndarray, optional): Item weights, defaults to WEIGHTS
    values (np.ndarray, optional): Item values, defaults to VALUES
    max_weight (float, optional): Maximum weight capacity, defaults to MAX_WEIGHT
    n_items (int, optional): Number of items, defaults to N_ITEMS

    Returns:
    object: Algorithm instance configured for knapsack problem
    """
    if weights is None:
        weights = WEIGHTS
    if values is None:
        values = VALUES
    if max_weight is None:
        max_weight = MAX_WEIGHT
    if n_items is None:
        n_items = N_ITEMS
    
    if algo_name == 'ABC':
        return ArtificialBeeColonyKnapsack(
            weights=weights,
            values=values,
            max_weight=max_weight,
            dim=n_items,
            num_employed_bees=pop_size,
            num_onlooker_bees=pop_size,
            max_iter=MAX_ITERATIONS,
            limit=30,
            seed=seed,
            verbose=VERBOSE
        )
    elif algo_name == 'FA':
        return FireflyKnapsack(
            weights=weights,
            values=values,
            max_weight=max_weight,
            population_size=pop_size,
            max_iter=MAX_ITERATIONS,
            seed=seed,
            verbose=VERBOSE
        )
    elif algo_name == 'Cuckoo':
        return CuckooSearchKnapsack(
            weights=weights,
            values=values,
            capacity=max_weight,
            dim=n_items,
            population_size=pop_size,
            max_iter=MAX_ITERATIONS,
            seed=seed,
            verbose=VERBOSE
        )
    elif algo_name == 'PSO':
        return ParitcleSwarmKnapsack(
            weights=weights,
            values=values,
            capacity=max_weight,
            dim=n_items,
            population_size=pop_size,
            max_iter=MAX_ITERATIONS,
            seed=seed,
            verbose=VERBOSE
        )
    elif algo_name == 'ACO':
        return AntColonyOptimizationKnapsack(
            weights=weights,
            values=values,
            capacity=max_weight,
            n_ants=pop_size,
            max_iter=MAX_ITERATIONS,
            seed=seed,
            verbose=VERBOSE
        )
    elif algo_name == 'HC':
        return HillClimbingKnapsack(
            weights=weights,
            values=values,
            capacity=max_weight,
            dim=n_items,
            max_iter=MAX_ITERATIONS,
            n_neighbors=pop_size,
            seed=seed,
            verbose=VERBOSE
        )
    elif algo_name == 'SA':
        return SimulatedAnnealingKnapsack(
            weights=weights,
            values=values,
            capacity=max_weight,
            dim=n_items,
            max_iter=MAX_ITERATIONS,
            step_size=0.5,
            initial_temp=100,
            seed=seed,
            verbose=VERBOSE
        )
    elif algo_name == 'GA':
        return GeneticAlgorithmKnapsack(
            weights=weights,
            values=values,
            capacity=max_weight,
            population_size=pop_size,
            max_iter=MAX_ITERATIONS,
            seed=seed,
            verbose=VERBOSE
        )
    else:
        raise ValueError(f"Unknown algorithm: {algo_name}")


def compute_solution_metrics(solution, values, weights):
    """Compute solution metrics including total value, weight, and binary representation.

    Parameters:
    solution (np.ndarray): Solution vector (binary or continuous)
    values (np.ndarray): Item values array
    weights (np.ndarray): Item weights array

    Returns:
    dict: Dictionary containing value, weight, and binary representation
    """
    binary = (solution > 0).astype(int) if solution.dtype != int else solution
    return {
        'value': np.sum(binary * values),
        'weight': np.sum(binary * weights),
        'binary': binary
    }


def run_algorithm(algo_name, pop_size, seed, weights=None, values=None, max_weight=None, n_items=None):
    """Run algorithm once and return performance metrics for knapsack problem.

    Parameters:
    algo_name (str): Name of the algorithm to run
    pop_size (int): Population size
    seed (int): Random seed for reproducibility
    weights (np.ndarray, optional): Item weights, defaults to WEIGHTS
    values (np.ndarray, optional): Item values, defaults to VALUES
    max_weight (float, optional): Maximum weight capacity, defaults to MAX_WEIGHT
    n_items (int, optional): Number of items, defaults to N_ITEMS

    Returns:
    dict: Dictionary containing value, weight, solution string, elapsed time, and space usage
    """
    np.random.seed(seed)
    random.seed(seed)
    start_time = time.time()
    if weights is None:
        weights = WEIGHTS
    if values is None:
        values = VALUES
    if max_weight is None:
        max_weight = MAX_WEIGHT
    if n_items is None:
        n_items = N_ITEMS
    
    algorithm = create_algorithm(algo_name, pop_size, seed, weights, values, max_weight, n_items)
    result = algorithm.run()
    
    solution, fitness, hist = result
    metrics = compute_solution_metrics(solution, values, weights)
    
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
    """Print formatted results table for knapsack optimization experiments.

    Parameters:
    results_list (list): List of dictionaries containing experiment results

    Returns:
    None
    """
    df = pd.DataFrame(results_list)
    has_test_case = 'Test Case' in df.columns
    if has_test_case:
        df = df.sort_values(by=['Test Case', 'Mean Value'], ascending=[True, False]).reset_index(drop=True)
    else:
        df = df.sort_values(by='Mean Value', ascending=False).reset_index(drop=True)
    
    df['Mean Value'] = df['Mean Value'].apply(lambda x: f"{x:.2f}")
    df['Mean Weight'] = df['Mean Weight'].apply(lambda x: f"{x:.2f}")
    df['Mean Time (s)'] = df['Mean Time (s)'].apply(lambda x: f"{x:.4f}")
    df['Mean Space (bytes)'] = df['Mean Space (bytes)'].apply(lambda x: f"{x:,.0f}")
    
    print("\n" + "=" * 100)
    print(" " * 32 + "Knapsack Optimization Results")
    print("=" * 100)
    
    col_widths = {
        'Test Case': 15,
        'Algorithm': 15,
        'Mean Value': 12,
        'Mean Weight': 12,
        'Mean Time (s)': 12,
        'Mean Space (bytes)': 20,
        'Solution': 30
    }
    
    if has_test_case:
        header = f"| {'Test Case':<{col_widths['Test Case']}} | {'Algorithm':<{col_widths['Algorithm']}} | {'Mean Value':<{col_widths['Mean Value']}} | {'Mean Weight':<{col_widths['Mean Weight']}} | {'Mean Time (s)':<{col_widths['Mean Time (s)']}} | {'Mean Space (bytes)':<{col_widths['Mean Space (bytes)']}} | {'Solution':<{col_widths['Solution']}} |"
        separator = "|" + "-" * (col_widths['Test Case'] + 2) + "|" + "-" * (col_widths['Algorithm'] + 2) + "|" + "-" * (col_widths['Mean Value'] + 2) + "|" + "-" * (col_widths['Mean Weight'] + 2) + "|" + "-" * (col_widths['Mean Time (s)'] + 2) + "|" + "-" * (col_widths['Mean Space (bytes)'] + 2) + "|" + "-" * (col_widths['Solution'] + 2) + "|"
    else:
        header = f"| {'Algorithm':<{col_widths['Algorithm']}} | {'Mean Value':<{col_widths['Mean Value']}} | {'Mean Weight':<{col_widths['Mean Weight']}} | {'Mean Time (s)':<{col_widths['Mean Time (s)']}} | {'Mean Space (bytes)':<{col_widths['Mean Space (bytes)']}} | {'Solution':<{col_widths['Solution']}} |"
        separator = "|" + "-" * (col_widths['Algorithm'] + 2) + "|" + "-" * (col_widths['Mean Value'] + 2) + "|" + "-" * (col_widths['Mean Weight'] + 2) + "|" + "-" * (col_widths['Mean Time (s)'] + 2) + "|" + "-" * (col_widths['Mean Space (bytes)'] + 2) + "|" + "-" * (col_widths['Solution'] + 2) + "|"
    
    print(header)
    print(separator)
    
    for _, row in df.iterrows():
        solution_str = str(row['Solution'])
        if len(solution_str) > col_widths['Solution']:
            solution_str = solution_str[:col_widths['Solution']-3] + "..."
        
        if has_test_case:
            line = f"| {str(row['Test Case']):<{col_widths['Test Case']}} | {str(row['Algorithm']):<{col_widths['Algorithm']}} | {str(row['Mean Value']):<{col_widths['Mean Value']}} | {str(row['Mean Weight']):<{col_widths['Mean Weight']}} | {str(row['Mean Time (s)']):<{col_widths['Mean Time (s)']}} | {str(row['Mean Space (bytes)']):<{col_widths['Mean Space (bytes)']}} | {solution_str:<{col_widths['Solution']}} |"
        else:
            line = f"| {str(row['Algorithm']):<{col_widths['Algorithm']}} | {str(row['Mean Value']):<{col_widths['Mean Value']}} | {str(row['Mean Weight']):<{col_widths['Mean Weight']}} | {str(row['Mean Time (s)']):<{col_widths['Mean Time (s)']}} | {str(row['Mean Space (bytes)']):<{col_widths['Mean Space (bytes)']}} | {solution_str:<{col_widths['Solution']}} |"
        print(line)
    
    print("=" * 100)
    print()


def run_experiments(test_cases=None):
    """Run knapsack optimization experiments for all algorithms and test cases.

    Parameters:
    test_cases (dict, optional): Dictionary of test cases to run, defaults to all TEST_CASES

    Returns:
    None
    """
    os.makedirs("results", exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    if test_cases is None:
        test_cases = TEST_CASES
    
    results = []
    print("\n--- Running Experiments ---")
    print(f"Number of test cases: {len(test_cases)}")
    print(f"Number of runs per test case: {N_RUNS}")
    print("=" * 100)
    for test_case_name, test_case in test_cases.items():
        print(f"\nRunning test case: {test_case_name}")
        print(f"  Items: {test_case['n_items']}, Max Weight: {test_case['max_weight']}")
        for algo in ALGOS:
            values, weights_list, solutions, times, spaces = [], [], [], [], []
            
            for pop in POP_SIZES:
                for run in range(N_RUNS):
                    seed = random.randint(0, 10000)
                    res = run_algorithm(
                        algo, 
                        pop, 
                        seed,
                        weights=test_case['weights'],
                        values=test_case['values'],
                        max_weight=test_case['max_weight'],
                        n_items=test_case['n_items']
                    )
                    values.append(res['value'])
                    weights_list.append(res['weight'])
                    solutions.append(res['solution'])
                    times.append(res['elapsed'])
                    spaces.append(res['space'])
            solution_str = str(solutions[0]) if solutions else ''
            if solution_str:
                solution_str = '\t' + solution_str
            
            results.append({
                "Test Case": test_case_name,
                "Algorithm": algo,
                'Mean Value': np.mean(values),
                'Mean Weight': np.mean(weights_list),
                'Solution': solution_str,
                'Mean Time (s)': np.mean(times),
                'Mean Space (bytes)': np.mean(spaces)
            })
            print(f"  {algo}: Mean Value = {np.mean(values):.2f}, Mean Time = {np.mean(times):.4f}s")
    
    print("\n" + "=" * 100)
    print_results(results)
    df = pd.DataFrame(results)
    df['Solution'] = df['Solution'].astype(str)
    df.to_csv(f"results/resultsKnapsack_{timestamp}.csv", index=False)
    print("Result CSV saved!")
    print("\nFinish")


if __name__ == "__main__":
    run_experiments()
