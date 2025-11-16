import pandas as pd
import matplotlib.pyplot as plt
import random
import time
import os
import sys
import numpy as np

from src.algorithms.swarm_algorithms.ACO import AntColonyOptimizationKnapsack
from src.problem.discrete.knapsack import WEIGHTS, VALUES, MAX_WEIGHT


FIXED_ALPHA = 1.0
FIXED_RHO = 0.5
FIXED_N_ANTS = 30
FIXED_Q = 1.0

PARAM_NAME_TO_TEST = 'beta'

PARAM_VALUES_TO_TEST = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]

N_RUNS = 10
N_ITERATIONS = 100

OUTPUT_IMAGE_FILE = f'ACOr_{PARAM_NAME_TO_TEST}_analysis.png'

def run_aco_instance(param_value_to_pass, run_seed):
    """Run ACO knapsack algorithm once and return convergence history.

    Parameters:
    param_value_to_pass (float): Parameter value to test for the algorithm
    run_seed (int): Random seed for reproducibility

    Returns:
    list: Convergence history as a list of fitness values per iteration
    """
    algo_parameters = {
        'weights': WEIGHTS,
        'values': VALUES,
        'capacity': MAX_WEIGHT,
        'n_ants': FIXED_N_ANTS,
        'max_iter': N_ITERATIONS,
        'seed': run_seed,
        'verbose': False,
        'alpha': FIXED_ALPHA,
        'beta': 1.0,
        'rho': FIXED_RHO,
        'Q': FIXED_Q
    }
    algo_parameters[PARAM_NAME_TO_TEST] = param_value_to_pass
    try:
        algorithm = AntColonyOptimizationKnapsack(**algo_parameters)
        _items, _best_value, hist = algorithm.run()
        return hist
    except NameError as e:
        print("\n[ERROR] Class 'AntColonyOptimizationKnapsack' not found.")
        print("=> DID YOU RUN CELL 1 (the cell above) FIRST?")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ALGORITHM ERROR] An error occurred during run: {e}")
        return [0] * N_ITERATIONS


def plot_and_save_results(results_df, param_name):
    """Plot convergence comparison for different parameter values and save to file.

    Parameters:
    results_df (pandas.DataFrame): DataFrame with convergence data, columns are parameter values, index is iteration
    param_name (str): Name of the parameter being tested

    Returns:
    None
    """
    print("\n--- Plotting and saving results... ---")
    plt.figure(figsize=(12, 7))
    ax = results_df.plot(
        figsize=(12, 7),
        grid=True,
        style='-o',
        markersize=3
    )
    ax.set_title(f'ACO Convergence Comparison for different "{param_name}" values')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Average Objective Function Value (Higher is better)')
    ax.legend(title=f'{param_name} Value', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    save_dir = 'src/visualization/parameter_analysis'
    os.makedirs(save_dir, exist_ok=True)
    img_save_path = os.path.join(save_dir, OUTPUT_IMAGE_FILE)
    plt.savefig(img_save_path, bbox_inches='tight')
    print(f"Saved plot to: {img_save_path}")
    plt.close()


def run_sensitivity_analysis():
    """Run sensitivity analysis for ACO knapsack parameter convergence.

    Parameters:
    None

    Returns:
    None
    """
    print(f"--- STARTING ACO CONVERGENCE ANALYSIS (KNAPSACK) ---")
    print(f"  Parameter under test: {PARAM_NAME_TO_TEST}")
    print(f"  Values to plot: {PARAM_VALUES_TO_TEST}")
    print(f"  Runs per value: {N_RUNS}\n")
    all_average_histories = {}
    start_total_time = time.time()
    for value in PARAM_VALUES_TO_TEST:
        histories_for_this_value = []
        start_value_time = time.time()
        display_value = round(value, 3)
        print(f"Testing {PARAM_NAME_TO_TEST} = {display_value}...")
        for i in range(N_RUNS):
            run_seed = random.randint(0, 1000000)
            history = run_aco_instance(value, run_seed)
            histories_for_this_value.append(history)
            print(f"  Run {i+1}/{N_RUNS}", end='\r')
        average_history = np.mean(histories_for_this_value, axis=0)
        all_average_histories[display_value] = average_history
        elapsed_time = time.time() - start_value_time
        print(" " * 60, end='\r')
        print(f"  -> Done! {PARAM_NAME_TO_TEST} = {display_value}: [Time: {elapsed_time:.2f}s]")
    print("\n--- All runs completed ---")
    total_time = time.time() - start_total_time
    print(f"Total execution time: {total_time:.2f} seconds")
    results_df = pd.DataFrame(all_average_histories)
    results_df.index.name = 'Iteration'
    print("\n--- AVERAGE CONVERGENCE DATA (SAMPLE) ---")
    print(results_df.tail())
    plot_and_save_results(results_df, PARAM_NAME_TO_TEST)
    print("\n--- Convergence analysis complete! ---")


if __name__ == "__main__":
    run_sensitivity_analysis()

