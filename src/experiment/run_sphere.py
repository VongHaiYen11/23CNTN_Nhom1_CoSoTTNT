import numpy as np
import time
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import random  # Để generate random seeds cho robustness

# ==== Import các thuật toán ====

from src.algorithms.swarm_algorithms.Cuckoo import CuckooSearch
from src.algorithms.swarm_algorithms.FA import FireflyAlgorithm  
from src.algorithms.swarm_algorithms.ABC import ArtificialBeeColony 
from src.algorithms.swarm_algorithms.PSO import ParticleSwarmOptimization
from src.algorithms.swarm_algorithms.ACO import AntColonyOptimizationContinuous

from src.algorithms.traditional_algorithms.GA import GeneticAlgorithmContinuous
from src.algorithms.traditional_algorithms.HC import HillClimbing
from src.algorithms.traditional_algorithms.SA import SimulatedAnnealing

# ==== Import bài toán ====
from src.problem.continuous.sphere import sphere  # Fitness func: min=0 at x=0

# ==== Cấu hình chung ====
N_RUNS = 5  # Nhỏ để test nhanh, tăng để accuracy mean/std

DIMS = [1, 10, 30]  # Test scalability với increasing dim
POP_SIZES = [50, 100, 300]  # Test scalability với increasing pop
MAX_ITERATIONS = 500
LOWER_BOUND = -5.12
UPPER_BOUND = 5.12
THRESHOLD = 1e-4  # Cho convergence speed (iterations to reach < threshold) và success (<1e-6)
OPTIMUM = np.zeros(1)  # For Sphere, optimum at 0, sẽ resize theo dim
SEAD = 42  # Seed cố định cho reproducibility

# ==== Cấu hình cho sensitivity analysis (nếu cần) ====
PARAM_VARIATIONS = {
    'FA': {'alpha': [0.1, 0.5, 1.0]},  # Vary alpha
    'ABC': {'limit': [20, 50, 100]},  # Vary limit
}

# List algos để so sánh (swarm + traditional, focus continuous Sphere)
ALGOS = ['FA', 'ABC', 'Cuckoo', 'PSO', 'HC', 'GA', 'SA', 'ACO'] 

def measure_space_usage(obj):
    if isinstance(obj, np.ndarray):
        return obj.nbytes + sys.getsizeof(obj)
    elif isinstance(obj, list):
        return sys.getsizeof(obj) + sum(measure_space_usage(o) for o in obj)
    return sys.getsizeof(obj)

def run_algorithm(algo_name, dim, pop_size, seed, param_vary=None):
    """
    Chạy 1 algo với config, trả metrics.
    Giải thích: Wrapper để chạy từng algo, capture time/space/history/pop cho metrics.
    param_vary: Dict để override param cho sensitivity (e.g. {'alpha': 0.5}).
    """
    np.random.seed(seed)
    random.seed(seed)
    start_time = time.time()
    
    if algo_name == 'FA':
        fa = FireflyAlgorithm(objective_function=sphere, lower_bound=LOWER_BOUND, upper_bound=UPPER_BOUND,
                              dimension=dim, max_iterations=MAX_ITERATIONS,
                              alpha=param_vary.get('alpha', 0.5) if param_vary else 0.5, seed=seed)
        best_sol, best_fit, hist = fa.run()
        population = fa.fireflies  
    
    elif algo_name == 'ABC':
        abc = ArtificialBeeColony(fitness_function=sphere, lower_bound=LOWER_BOUND, upper_bound=UPPER_BOUND,
                                  problem_size=dim, num_employed_bees=pop_size, num_onlooker_bees=pop_size,
                                  max_iterations=MAX_ITERATIONS, limit=50, seed=seed)
        best_sol, best_fit, hist = abc.run()
        population = abc.food_sources  # Add self.food_sources nếu cần
    
    elif algo_name == 'Cuckoo':
        cs = CuckooSearch(fitness_func=sphere, lower_bound=LOWER_BOUND, upper_bound=UPPER_BOUND,
                          dim=dim, population_size=pop_size, max_iter=MAX_ITERATIONS, seed=seed)
        best_sol, best_fit, hist = cs.run()  # Note: Cuckoo không trả hist/pop, add nếu cần (hist=[] in class)
        hist = [best_fit] * MAX_ITERATIONS  # Placeholder nếu không có
        population = None
    
    elif algo_name == 'PSO':
        best_sol, best_fit, hist = ParticleSwarmOptimization(
            objective_function=sphere,
            lower_bound=LOWER_BOUND,
            upper_bound=UPPER_BOUND,
            dim=dim,
            population_size=pop_size,
            max_iter=MAX_ITERATIONS,
            seed=seed
        ).run()
        
        hist = [best_fit] * MAX_ITERATIONS  # placeholder, vì class chưa trả history
        population = None  # PSO có particles, add nếu implement

    elif algo_name == 'ACO':
        best_sol, best_fit, hist = AntColonyOptimizationContinuous(
            objective_function=sphere,
            lower_bound=LOWER_BOUND,
            upper_bound=UPPER_BOUND,
            dim=dim,
            max_iter=MAX_ITERATIONS,
            seed=seed
        ).run()
        hist = [best_fit] * MAX_ITERATIONS  # No hist, placeholder
        population = None

    elif algo_name == 'HC':
        best_sol, best_fit, hist = HillClimbing(
            fitness_func=sphere,
            lower_bound=LOWER_BOUND,
            upper_bound=UPPER_BOUND,
            dim=dim,
            max_iter=MAX_ITERATIONS,
            seed=seed
        ).run()
        hist = [best_fit] * MAX_ITERATIONS  # No hist, placeholder
        population = None

    elif algo_name == 'GA':
        best_sol, best_fit, hist = GeneticAlgorithmContinuous(
            fitness_func=sphere,
            lower_bound=LOWER_BOUND,
            upper_bound=UPPER_BOUND,
            dim=dim,
            population_size=pop_size,
            max_iter=MAX_ITERATIONS,
            seed=seed
        ).run()
        hist = [best_fit] * MAX_ITERATIONS
        population = None

    elif algo_name == 'SA':
        best_sol, best_fit, hist = SimulatedAnnealing(
            fitness_func=sphere,
            lower_bound=LOWER_BOUND,
            upper_bound=UPPER_BOUND,
            dim=dim,
            max_iter=MAX_ITERATIONS * 100,  # SA cần nhiều iter
            seed=seed
        ).run()
        hist = [best_fit] * MAX_ITERATIONS
        population = None

    
    elapsed = time.time() - start_time
    space = measure_space_usage(population) if population is not None else measure_space_usage(best_sol)

    return {
        'best_fit': best_fit, 'elapsed': elapsed, 'space': space
    } 


def run_experiments():
    """Chạy full experiments, collect data cho all metrics.
    Giải thích: Loop qua dims/pop_sizes (scalability), trong mỗi config loop N_runs với random seeds (robustness).
    Cho sensitivity: Riêng loop vary params. Save CSVs và plots."""
    os.makedirs("results", exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    
    result = []
    print("\n--- Measuring  ---")
    for dim in DIMS:
        for algo in ALGOS:
            fits, times, spaces = [], [], []
            i=0
            for pop in POP_SIZES:
                print(i)
                i+=1
                for _ in range(N_RUNS):
                    seed = random.randint(0, SEAD) 
                    res = run_algorithm(algo, dim, pop, seed=seed)
                    fits.append(res['best_fit'])
                    times.append(res['elapsed'])
                    spaces.append(res['space'])
            
            result.append({
                "DIM": dim,
                "Algorithm": algo,
                'Mean Fitness': np.mean(fits),
                'Mean Time (s)': np.mean(times),
                'Mean Space (bytes)': np.mean(spaces)
            })

    
    
    pd.DataFrame(result).to_csv(f"results/resultsDIM_{timestamp}.csv", index=False)
    print("Result CSV saved!")
    
    print("\nFinish")

if __name__ == "__main__":
    run_experiments()