import numpy as np
import time
import os
import pandas as pd
import sys
import random
import matplotlib.pyplot as plt

from src.algorithms.swarm_algorithms.Cuckoo import CuckooSearch
from src.algorithms.swarm_algorithms.FA import FireflyAlgorithm
from src.algorithms.swarm_algorithms.ABC import ArtificialBeeColony
from src.algorithms.swarm_algorithms.PSO import ParticleSwarmOptimization
from src.algorithms.swarm_algorithms.ACO import AntColonyOptimizationContinuous
from src.problem.continuous.sphere import sphere

# Experiment settings
N_RUNS = 1  # 1 nên là besfitness, hiện mean_fitness nhưng mà mean của 1 thì là best
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
        # Run once (N_RUNS = 1)
        seed = SEED
        best_fit = run_parameter_experiment(algo_name, param_name, param_value, seed)
        
        if best_fit is not None:
            results.append({
                'algorithm_name': algo_name,
                'parameter_name': param_name,
                'parameter_value': param_value,
                'best_fitness': best_fit
            })
            print(f"  {param_name}={param_value:.4f}: Best Fitness = {best_fit:.6e}")
    
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


def plot_parameter_analysis(df, save_dir='src/visualization/parameter'):
    """
    Generate visualization plots for each hyperparameter.
    
    Args:
        df: DataFrame containing parameter analysis results
        save_dir: Directory to save the plots (default: src/visualization/parameter)
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Color palette for different algorithms
    colors = {
        'FA': '#FF6B6B',
        'ABC': '#4ECDC4',
        'Cuckoo': '#45B7D1',
        'PSO': '#FFA07A',
        'ACO': '#85C1E2'
    }
    
    # Get unique algorithm-parameter combinations
    unique_combinations = df[['algorithm_name', 'parameter_name']].drop_duplicates()
    
    print("\n" + "=" * 80)
    print("Generating Parameter Analysis Visualizations")
    print("=" * 80)
    
    for _, row in unique_combinations.iterrows():
        algo_name = row['algorithm_name']
        param_name = row['parameter_name']
        
        # Filter data for this algorithm and parameter
        param_data = df[(df['algorithm_name'] == algo_name) & 
                       (df['parameter_name'] == param_name)].copy()
        
        # Sort by parameter value
        param_data = param_data.sort_values('parameter_value')
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot line with markers
        color = colors.get(algo_name, '#95A5A6')
        ax.plot(param_data['parameter_value'], param_data['best_fitness'], 
               marker='o', linewidth=2.5, markersize=8, color=color, 
               label=algo_name, alpha=0.8)
        
        # Add grid
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Set labels and title
        ax.set_xlabel(f'{param_name}', fontsize=13, fontweight='bold')
        ax.set_ylabel('Best Fitness', fontsize=13, fontweight='bold')
        ax.set_title(f'Parameter Analysis: {algo_name} - {param_name}', 
                    fontsize=14, fontweight='bold', pad=15)
        
        # Add padding to y-axis
        y_min = param_data['best_fitness'].min()
        y_max = param_data['best_fitness'].max()
        
        # Use log scale for y-axis if fitness values span large range
        use_log_scale = False
        if y_min > 0:
            fitness_ratio = y_max / y_min
            if fitness_ratio > 100:
                ax.set_yscale('log')
                use_log_scale = True
        
        # Add legend
        ax.legend(loc='best', fontsize=11, framealpha=0.9)
        
        # Set y-axis limits
        if use_log_scale:
            ax.set_ylim(y_min / 1.5, y_max * 1.5)
        else:
            y_padding = (y_max - y_min) * 0.15 if y_max > y_min else y_max * 0.15
            ax.set_ylim(max(0, y_min - y_padding), y_max + y_padding)
        
        plt.tight_layout()
        
        # Save plot
        filename = f'{algo_name}_{param_name}_analysis.png'
        save_path = os.path.join(save_dir, filename)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {filename}")
    
    print("=" * 80)
    print("All visualizations generated successfully!")
    print("=" * 80)


def visualize_parameter_analysis_from_csv(csv_path, save_dir='src/visualization/parameter'):
    """
    Generate visualizations from an existing parameter analysis CSV file.
    
    Args:
        csv_path: Path to the parameter analysis CSV file
        save_dir: Directory to save the plots (default: src/visualization/parameter)
    """
    df = pd.read_csv(csv_path)
    
    # Handle both old and new column names for backward compatibility
    if 'mean_fitness' in df.columns and 'best_fitness' not in df.columns:
        df['best_fitness'] = df['mean_fitness']
    
    plot_parameter_analysis(df, save_dir)


if __name__ == "__main__":
    df = run_all_parameter_analysis()
    
    # Generate visualizations
    print("\n")
    plot_parameter_analysis(df)

