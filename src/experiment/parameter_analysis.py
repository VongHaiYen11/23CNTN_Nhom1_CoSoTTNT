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
from src.problem.continuous.sphere import sphere

# Experiment settings
N_RUNS = 1  # Number of runs per parameter value
DIM = 30
POP_SIZE = 50
MAX_ITERATIONS = 200
LOWER_BOUND = -5.12
UPPER_BOUND = 5.12
SEED = 42
VERBOSE = False

# Swarm algorithms to test
SWARM_ALGOS = ['FA', 'ABC', 'Cuckoo', 'PSO', 'ACO']

# Parameter ranges (10 values each)
PARAMETER_RANGES = {
    'PSO': {
        'w': np.linspace(0.1, 1.0, 10),  # Inertia weight
        'c1': np.linspace(0.5, 2.5, 10),  # Cognitive coefficient
        'c2': np.linspace(0.5, 2.5, 10),  # Social coefficient
    },
    'ABC': {
        'limit': np.linspace(10, 100, 10, dtype=int),  # Abandonment limit
    },
    'FA': {
        'alpha': np.linspace(0.1, 1.0, 10),  # Randomness parameter
        'beta0': np.linspace(0.5, 2.0, 10),  # Attractiveness at distance 0
        'gamma': np.linspace(0.001, 0.1, 10),  # Absorption coefficient
    },
    'Cuckoo': {
        'pa': np.linspace(0.1, 0.5, 10),  # Discovery rate
        'alpha': np.linspace(0.001, 0.1, 10),  # Step size
        'beta': np.linspace(1.0, 2.0, 10),  # Levy flight parameter
    },
    'ACO': {
        'rho': np.linspace(0.5, 0.99, 10),  # Evaporation rate
    }
}


def create_algorithm_with_parameter(algo_name, param_name, param_value, seed):
    """Create algorithm instance with a specific parameter value"""
    if algo_name == 'FA':
        if param_name == 'alpha':
            return FireflyAlgorithm(
                fitness_func=sphere,
                lower_bound=LOWER_BOUND,
                upper_bound=UPPER_BOUND,
                dim=DIM,
                population_size=POP_SIZE,
                max_iter=MAX_ITERATIONS,
                alpha=param_value,
                beta0=1.0,  # Default
                gamma=0.01,  # Default
                seed=seed,
                verbose=VERBOSE
            )
        elif param_name == 'beta0':
            return FireflyAlgorithm(
                fitness_func=sphere,
                lower_bound=LOWER_BOUND,
                upper_bound=UPPER_BOUND,
                dim=DIM,
                population_size=POP_SIZE,
                max_iter=MAX_ITERATIONS,
                alpha=0.5,  # Default
                beta0=param_value,
                gamma=0.01,  # Default
                seed=seed,
                verbose=VERBOSE
            )
        elif param_name == 'gamma':
            return FireflyAlgorithm(
                fitness_func=sphere,
                lower_bound=LOWER_BOUND,
                upper_bound=UPPER_BOUND,
                dim=DIM,
                population_size=POP_SIZE,
                max_iter=MAX_ITERATIONS,
                alpha=0.5,  # Default
                beta0=1.0,  # Default
                gamma=param_value,
                seed=seed,
                verbose=VERBOSE
            )
    
    elif algo_name == 'ABC':
        if param_name == 'limit':
            return ArtificialBeeColony(
                fitness_func=sphere,
                lower_bound=LOWER_BOUND,
                upper_bound=UPPER_BOUND,
                dim=DIM,
                num_employed_bees=POP_SIZE,
                num_onlooker_bees=POP_SIZE,
                max_iter=MAX_ITERATIONS,
                limit=param_value,
                seed=seed,
                verbose=VERBOSE
            )
    
    elif algo_name == 'Cuckoo':
        if param_name == 'pa':
            return CuckooSearch(
                fitness_func=sphere,
                lower_bound=LOWER_BOUND,
                upper_bound=UPPER_BOUND,
                dim=DIM,
                population_size=POP_SIZE,
                max_iter=MAX_ITERATIONS,
                pa=param_value,
                alpha=0.01,  # Default
                beta=1.5,  # Default
                seed=seed,
                verbose=VERBOSE
            )
        elif param_name == 'alpha':
            return CuckooSearch(
                fitness_func=sphere,
                lower_bound=LOWER_BOUND,
                upper_bound=UPPER_BOUND,
                dim=DIM,
                population_size=POP_SIZE,
                max_iter=MAX_ITERATIONS,
                pa=0.25,  # Default
                alpha=param_value,
                beta=1.5,  # Default
                seed=seed,
                verbose=VERBOSE
            )
        elif param_name == 'beta':
            return CuckooSearch(
                fitness_func=sphere,
                lower_bound=LOWER_BOUND,
                upper_bound=UPPER_BOUND,
                dim=DIM,
                population_size=POP_SIZE,
                max_iter=MAX_ITERATIONS,
                pa=0.25,  # Default
                alpha=0.01,  # Default
                beta=param_value,
                seed=seed,
                verbose=VERBOSE
            )
    
    elif algo_name == 'PSO':
        if param_name == 'w':
            return ParticleSwarmOptimization(
                fitness_func=sphere,
                lower_bound=LOWER_BOUND,
                upper_bound=UPPER_BOUND,
                dim=DIM,
                population_size=POP_SIZE,
                max_iter=MAX_ITERATIONS,
                w=param_value,
                c1=1.5,  # Default
                c2=1.5,  # Default
                seed=seed,
                verbose=VERBOSE
            )
        elif param_name == 'c1':
            return ParticleSwarmOptimization(
                fitness_func=sphere,
                lower_bound=LOWER_BOUND,
                upper_bound=UPPER_BOUND,
                dim=DIM,
                population_size=POP_SIZE,
                max_iter=MAX_ITERATIONS,
                w=0.7,  # Default
                c1=param_value,
                c2=1.5,  # Default
                seed=seed,
                verbose=VERBOSE
            )
        elif param_name == 'c2':
            return ParticleSwarmOptimization(
                fitness_func=sphere,
                lower_bound=LOWER_BOUND,
                upper_bound=UPPER_BOUND,
                dim=DIM,
                population_size=POP_SIZE,
                max_iter=MAX_ITERATIONS,
                w=0.7,  # Default
                c1=1.5,  # Default
                c2=param_value,
                seed=seed,
                verbose=VERBOSE
            )
    
    elif algo_name == 'ACO':
        if param_name == 'rho':
            return AntColonyOptimizationContinuous(
                fitness_func=sphere,
                dim=DIM,
                lower_bound=LOWER_BOUND,
                upper_bound=UPPER_BOUND,
                population_size=POP_SIZE,
                max_iter=MAX_ITERATIONS,
                rho=param_value,
                seed=seed,
                verbose=VERBOSE
            )
    
    raise ValueError(f"Unknown algorithm or parameter: {algo_name}, {param_name}")


def run_parameter_experiment(algo_name, param_name, param_value, seed):
    """Run algorithm with specific parameter value and return best fitness"""
    np.random.seed(seed)
    random.seed(seed)
    
    try:
        algorithm = create_algorithm_with_parameter(algo_name, param_name, param_value, seed)
        result = algorithm.run()
        best_sol, best_fit, hist = result
        return best_fit
    except Exception as e:
        print(f"  Error: {e}")
        return None


def analyze_parameter(algo_name, param_name, param_values):
    """Analyze one parameter for one algorithm"""
    print(f"\nAnalyzing {algo_name} - Parameter: {param_name}")
    print(f"  Testing {len(param_values)} values: {param_values[0]:.4f} to {param_values[-1]:.4f}")
    
    results = []
    
    for param_value in param_values:
        fitnesses = []
        
        for run in range(N_RUNS):
            seed = SEED + run  # Different seed for each run
            best_fit = run_parameter_experiment(algo_name, param_name, param_value, seed)
            if best_fit is not None:
                fitnesses.append(best_fit)
        
        if fitnesses:
            mean_fitness = np.mean(fitnesses)
            std_fitness = np.std(fitnesses)
            results.append({
                'algorithm_name': algo_name,
                'parameter_name': param_name,
                'parameter_value': param_value,
                'best_fitness': mean_fitness,
                'std_fitness': std_fitness,
                'min_fitness': np.min(fitnesses),
                'max_fitness': np.max(fitnesses)
            })
            print(f"  {param_name}={param_value:.4f}: Mean Fitness = {mean_fitness:.6e}")
    
    return results


def run_all_parameter_analysis():
    """Run parameter analysis for all swarm algorithms"""
    os.makedirs("results", exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    
    all_results = []
    
    print("=" * 80)
    print("Parameter Analysis for Swarm Algorithms")
    print("=" * 80)
    print(f"Dimension: {DIM}")
    print(f"Population Size: {POP_SIZE}")
    print(f"Max Iterations: {MAX_ITERATIONS}")
    print(f"Number of runs per parameter value: {N_RUNS}")
    print("=" * 80)
    
    # Analyze each algorithm and its parameters
    for algo_name in SWARM_ALGOS:
        if algo_name in PARAMETER_RANGES:
            for param_name, param_values in PARAMETER_RANGES[algo_name].items():
                results = analyze_parameter(algo_name, param_name, param_values)
                all_results.extend(results)
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(all_results)
    
    # Save full results
    full_csv_path = f"results/parameter_analysis_full_{timestamp}.csv"
    df.to_csv(full_csv_path, index=False)
    print(f"\nFull results saved to: {full_csv_path}")
    
    # Save simplified results (only algorithm_name, best_fitness, parameter_name, parameter_value)
    simplified_df = df[['algorithm_name', 'best_fitness', 'parameter_name', 'parameter_value']]
    simplified_csv_path = f"results/parameter_analysis_{timestamp}.csv"
    simplified_df.to_csv(simplified_csv_path, index=False)
    print(f"Simplified results saved to: {simplified_csv_path}")
    
    # Print summary
    print("\n" + "=" * 80)
    print("Summary by Algorithm and Parameter")
    print("=" * 80)
    
    for algo_name in SWARM_ALGOS:
        if algo_name in PARAMETER_RANGES:
            print(f"\n{algo_name}:")
            for param_name in PARAMETER_RANGES[algo_name].keys():
                algo_param_df = df[(df['algorithm_name'] == algo_name) & 
                                   (df['parameter_name'] == param_name)]
                if not algo_param_df.empty:
                    best_row = algo_param_df.loc[algo_param_df['best_fitness'].idxmin()]
                    print(f"  {param_name}:")
                    print(f"    Best value: {best_row['parameter_value']:.4f}")
                    print(f"    Best fitness: {best_row['best_fitness']:.6e}")
    
    print("\n" + "=" * 80)
    print("Analysis complete!")
    print("=" * 80)
    
    return df


if __name__ == "__main__":
    run_all_parameter_analysis()

