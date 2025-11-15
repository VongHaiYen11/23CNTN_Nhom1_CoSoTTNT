import numpy as np
import time
import os
import pandas as pd
import sys
import random

from src.algorithms.swarm_algorithms.Cuckoo import CuckooSearch
from src.algorithms.swarm_algorithms.FA import FireflyAlgorithm
from src.algorithms.swarm_algorithms.ABC import ArtificialBeeColony
from src.algorithms.swarm_algorithms.PSO import ParticleSwarmOptimization
from src.algorithms.swarm_algorithms.ACO import AntColonyOptimizationContinuous
from src.algorithms.traditional_algorithms.GA import GeneticAlgorithmContinuous
from src.algorithms.traditional_algorithms.HC import HillClimbing
from src.algorithms.traditional_algorithms.SA import SimulatedAnnealing
from src.problem.continuous.sphere import sphere

N_RUNS = 1
DIMS = [30]
POP_SIZES = [50]
MAX_ITERATIONS = 100
LOWER_BOUND = -5.12
UPPER_BOUND = 5.12
SEED = 42
VERBOSE = True

ALGOS = ['FA', 'ABC', 'Cuckoo', 'PSO', 'HC', 'GA', 'SA', 'ACO']


def measure_space_usage(obj):
    if isinstance(obj, np.ndarray):
        return obj.nbytes + sys.getsizeof(obj)
    elif isinstance(obj, list):
        return sys.getsizeof(obj) + sum(measure_space_usage(o) for o in obj)
    return sys.getsizeof(obj)


def create_algorithm(algo_name, dim, pop_size, seed):
    if algo_name == 'FA':
        return FireflyAlgorithm(
            fitness_func=sphere,
            lower_bound=LOWER_BOUND,
            upper_bound=UPPER_BOUND,
            dim=dim,
            population_size=pop_size,
            max_iter=MAX_ITERATIONS,
            alpha=0.5,
            seed=seed,
            verbose=VERBOSE
        )
    elif algo_name == 'ABC':
        return ArtificialBeeColony(
            fitness_func=sphere,
            lower_bound=LOWER_BOUND,
            upper_bound=UPPER_BOUND,
            dim=dim,
            num_employed_bees=pop_size,
            num_onlooker_bees=pop_size,
            max_iter=MAX_ITERATIONS,
            limit=50,
            seed=seed,
            verbose=VERBOSE
        )
    elif algo_name == 'Cuckoo':
        return CuckooSearch(
            fitness_func=sphere,
            lower_bound=LOWER_BOUND,
            upper_bound=UPPER_BOUND,
            dim=dim,
            population_size=pop_size,
            max_iter=MAX_ITERATIONS,
            seed=seed,
            verbose=VERBOSE
        )
    elif algo_name == 'PSO':
        return ParticleSwarmOptimization(
            fitness_func=sphere,
            lower_bound=LOWER_BOUND,
            upper_bound=UPPER_BOUND,
            dim=dim,
            population_size=pop_size,
            max_iter=MAX_ITERATIONS,
            seed=seed,
            verbose=VERBOSE
        )
    elif algo_name == 'ACO':
        return AntColonyOptimizationContinuous(
            fitness_func=sphere,
            dim=dim,
            lower_bound=LOWER_BOUND,
            upper_bound=UPPER_BOUND,
            population_size=pop_size,
            max_iter=MAX_ITERATIONS,
            seed=seed,
            verbose=VERBOSE
        )
    elif algo_name == 'HC':
        return HillClimbing(
            fitness_func=sphere,
            lower_bound=LOWER_BOUND,
            upper_bound=UPPER_BOUND,
            dim=dim,
            max_iter=MAX_ITERATIONS,
            seed=seed,
            verbose=VERBOSE
        )
    elif algo_name == 'GA':
        return GeneticAlgorithmContinuous(
            fitness_func=sphere,
            lower_bound=LOWER_BOUND,
            upper_bound=UPPER_BOUND,
            dim=dim,
            population_size=pop_size,
            max_iter=MAX_ITERATIONS,
            seed=seed,
            verbose=VERBOSE
        )
    elif algo_name == 'SA':
        return SimulatedAnnealing(
            fitness_func=sphere,
            lower_bound=LOWER_BOUND,
            upper_bound=UPPER_BOUND,
            dim=dim,
            max_iter=MAX_ITERATIONS,
            seed=seed,
            verbose=VERBOSE
        )
    else:
        raise ValueError(f"Unknown algorithm: {algo_name}")


def run_algorithm(algo_name, dim, pop_size, seed):
    np.random.seed(seed)
    random.seed(seed)
    start_time = time.time()
    
    algorithm = create_algorithm(algo_name, dim, pop_size, seed)
    result = algorithm.run()
    
    if algo_name == 'Cuckoo':
        best_sol, best_fit = result
        hist = [best_fit] * MAX_ITERATIONS
        population = None
    elif algo_name == 'GA':
        best_sol, best_fit, hist, population = result
    elif algo_name == 'FA':
        best_sol, best_fit, hist = result
        population = algorithm.fireflies
    elif algo_name == 'ABC':
        best_sol, best_fit, hist = result
        population = algorithm.food_sources
    else:
        best_sol, best_fit, hist = result
        population = None
    
    elapsed = time.time() - start_time
    space = measure_space_usage(population) if population is not None else measure_space_usage(best_sol)
    
    return {
        'best_fit': best_fit,
        'elapsed': elapsed,
        'space': space
    }


def print_results(results_list):
    df = pd.DataFrame(results_list)
    
    df['Mean Fitness'] = df['Mean Fitness'].apply(lambda x: f"{x:.6f}")
    df['Mean Time (s)'] = df['Mean Time (s)'].apply(lambda x: f"{x:.4f}")
    df['Mean Space (bytes)'] = df['Mean Space (bytes)'].apply(lambda x: f"{x:,.0f}")
    
    print("\n" + "=" * 95)
    print(" " * 32 + "Sphere Optimization Results")
    print("=" * 95)
    
    col_widths = {
        'DIM': 8,
        'Algorithm': 12,
        'Mean Fitness': 15,
        'Mean Time (s)': 15,
        'Mean Space (bytes)': 20
    }
    
    header = f"| {'DIM':<{col_widths['DIM']}} | {'Algorithm':<{col_widths['Algorithm']}} | {'Mean Fitness':<{col_widths['Mean Fitness']}} | {'Mean Time (s)':<{col_widths['Mean Time (s)']}} | {'Mean Space (bytes)':<{col_widths['Mean Space (bytes)']}} |"
    separator = "|" + "-" * (col_widths['DIM'] + 2) + "|" + "-" * (col_widths['Algorithm'] + 2) + "|" + "-" * (col_widths['Mean Fitness'] + 2) + "|" + "-" * (col_widths['Mean Time (s)'] + 2) + "|" + "-" * (col_widths['Mean Space (bytes)'] + 2) + "|"
    
    print(header)
    print(separator)
    
    for _, row in df.iterrows():
        line = f"| {str(row['DIM']):<{col_widths['DIM']}} | {str(row['Algorithm']):<{col_widths['Algorithm']}} | {str(row['Mean Fitness']):<{col_widths['Mean Fitness']}} | {str(row['Mean Time (s)']):<{col_widths['Mean Time (s)']}} | {str(row['Mean Space (bytes)']):<{col_widths['Mean Space (bytes)']}} |"
        print(line)
    
    print("=" * 95)
    print()


def run_experiments():
    os.makedirs("results", exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    
    results = []
    print("\n--- Running Experiments ---")
    
    for dim in DIMS:
        for algo in ALGOS:
            fits, times, spaces = [], [], []
            
            for pop in POP_SIZES:
                for _ in range(N_RUNS):
                    seed = random.randint(0, SEED)
                    res = run_algorithm(algo, dim, pop, seed)
                    fits.append(res['best_fit'])
                    times.append(res['elapsed'])
                    spaces.append(res['space'])
            
            results.append({
                "DIM": dim,
                "Algorithm": algo,
                'Mean Fitness': np.mean(fits),
                'Mean Time (s)': np.mean(times),
                'Mean Space (bytes)': np.mean(spaces)
            })
    
    print_results(results)
    pd.DataFrame(results).to_csv(f"results/resultsDIM_{timestamp}.csv", index=False)
    print("Result CSV saved!")
    print("\nFinish")


if __name__ == "__main__":
    run_experiments()
