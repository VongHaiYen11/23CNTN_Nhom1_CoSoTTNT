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
# Để robustness: Số runs với random seeds khác nhau (tăng lên 30 cho report chính thức)
N_RUNS = 5  # Nhỏ để test nhanh, tăng để accuracy mean/std
# Để scalability: Vary DIM (problem size) và POP_SIZES
DIMS = [10, 30, 100]  # Test scalability với increasing dim
POP_SIZES = [50, 100, 200]  # Test scalability với increasing pop
MAX_ITERATIONS = 300
LOWER_BOUND = -5.12
UPPER_BOUND = 5.12
THRESHOLD = 1e-4  # Cho convergence speed (iterations to reach < threshold) và success (<1e-6)
OPTIMUM = np.zeros(1)  # For Sphere, optimum at 0, sẽ resize theo dim

# Để parameter sensitivity: Dict params vary cho từng algo (chỉ vary 1-2 key param để đơn giản)
PARAM_VARIATIONS = {
    'FA': {'alpha': [0.1, 0.5, 1.0]},  # Vary alpha
    'ABC': {'limit': [20, 50, 100]},  # Vary limit
    # Add for others if needed, e.g. PSO: {'c1': [1.0, 2.0]}
}

# List algos để so sánh (swarm + traditional, focus continuous Sphere)
ALGOS = ['FA', 'ABC', 'Cuckoo', 'PSO', 'HC', 'GA', 'SA', 'ACO']  # Add ACO/CS nếu implement

def measure_space_usage(obj):
    """Ước lượng space complexity (bytes) bằng sys.getsizeof, recursive cho arrays/lists.
    Giải thích: Để so sánh space, đo size của population/food_sources (O(pop*dim) cho swarm).
    Traditional như HC/SA chỉ O(dim), nên tiết kiệm hơn."""
    if isinstance(obj, np.ndarray):
        return obj.nbytes  # Accurate cho NumPy arrays
    elif isinstance(obj, list):
        return sum(sys.getsizeof(o) for o in obj)
    return sys.getsizeof(obj)

def calculate_convergence_speed(history, threshold=THRESHOLD):
    """Tính số iterations để đạt < threshold (convergence speed).
    Giải thích: Từ fitness_history, tìm min iter nơi best_fit < threshold; nếu không đạt, return max_iter."""
    for i, fit in enumerate(history):
        if fit < threshold:
            return i + 1  # 1-based
    return len(history)  # Không đạt

def calculate_diversity(population):
    """Tính diversity: mean std qua dimensions (np.std(pop, axis=0).mean()).
    Giải thích: Đo exploration/exploitation; swarm cao diversity ban đầu, giảm dần."""
    if population is None:
        return 0.0  # Traditional không có pop
    return np.mean(np.std(population, axis=0))

def calculate_solution_quality(best_sol, optimum):
    """Tính distance to optimum: np.linalg.norm(best_sol - optimum).
    Giải thích: Đo accuracy ngoài best_fit, đặc biệt nếu optimum known như Sphere."""
    return np.linalg.norm(best_sol - optimum)

def run_algorithm(algo_name, dim, pop_size, seed, param_vary=None):
    """Chạy 1 algo với config, trả metrics.
    Giải thích: Wrapper để chạy từng algo, capture time/space/history/pop cho metrics.
    param_vary: Dict để override param cho sensitivity (e.g. {'alpha': 0.5})."""
    np.random.seed(seed)
    random.seed(seed)
    start_time = time.time()
    
    if algo_name == 'FA':
        fa = FireflyAlgorithm(objective_function=sphere, lower_bound=LOWER_BOUND, upper_bound=UPPER_BOUND,
                              dimension=dim, population_size=pop_size, max_iterations=MAX_ITERATIONS,
                              alpha=param_vary.get('alpha', 0.5) if param_vary else 0.5, seed=seed)
        best_sol, best_fit, hist = fa.run()
        population = fa.fireflies  # Giả sử class có attr fireflies (add nếu cần: self.fireflies sau init)
    
    elif algo_name == 'ABC':
        abc = ArtificialBeeColony(fitness_function=sphere, lower_bound=LOWER_BOUND, upper_bound=UPPER_BOUND,
                                  problem_size=dim, num_employed_bees=pop_size, num_onlooker_bees=pop_size,
                                  max_iterations=MAX_ITERATIONS, limit=param_vary.get('limit', 50) if param_vary else 50, seed=seed)
        best_sol, best_fit, hist = abc.run()
        population = abc.food_sources  # Add self.food_sources nếu cần
    
    elif algo_name == 'Cuckoo':
        cs = CuckooSearch(fitness_func=sphere, lower_bound=LOWER_BOUND, upper_bound=UPPER_BOUND,
                          dim=dim, population_size=pop_size, max_iter=MAX_ITERATIONS, seed=seed)
        best_sol, best_fit = cs.run()  # Note: Cuckoo không trả hist/pop, add nếu cần (hist=[] in class)
        hist = [best_fit] * MAX_ITERATIONS  # Placeholder nếu không có
        population = None
    
    elif algo_name == 'PSO':
        pso = best_sol, best_fit = ParticleSwarmOptimization(
            objective_function=sphere,
            lower_bound=LOWER_BOUND,
            upper_bound=UPPER_BOUND,
            dim=dim,
            population_size=pop_size,
            max_iter=MAX_ITERATIONS,
            seed=seed
        ).run()
        best_sol, best_fit, hist = pso.run()
        hist = [best_fit] * MAX_ITERATIONS  # placeholder, vì class chưa trả history
        population = None  # PSO có particles, add nếu implement

    elif algo_name == 'ACO':
        aco = best_sol, best_fit = AntColonyOptimizationContinuous(
            fitness_func=sphere,
            lower_bound=LOWER_BOUND,
            upper_bound=UPPER_BOUND,
            dim=dim,
            max_iter=MAX_ITERATIONS,
            seed=seed
        ).run()
        best_sol, best_fit, hist = aco.run()
        population = None

    elif algo_name == 'HC':
        best_sol, best_fit = HillClimbing(
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
        best_sol, best_fit = GeneticAlgorithmContinuous(
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
        best_sol, best_fit = SimulatedAnnealing(
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
    conv_speed = calculate_convergence_speed(hist)
    diversity = calculate_diversity(population)
    quality = calculate_solution_quality(best_sol, np.zeros(dim))
    success = 1 if best_fit < 1e-6 else 0
    
    return {
        'best_fit': best_fit, 'elapsed': elapsed, 'space': space, 'conv_speed': conv_speed,
        'diversity': diversity, 'quality': quality, 'success': success, 'hist': hist
    }


def run_experiments():
    """Chạy full experiments, collect data cho all metrics.
    Giải thích: Loop qua dims/pop_sizes (scalability), trong mỗi config loop N_runs với random seeds (robustness).
    Cho sensitivity: Riêng loop vary params. Save CSVs và plots."""
    os.makedirs("results", exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    
    # 1. Robustness & Basic Metrics: Fix dim=30, pop=100, vary seeds
    robustness_data = []
    print("\n--- Measuring Robustness (Multiple Runs, dim=30, pop=100) ---")
    for algo in ALGOS:
        fits, times, spaces, convs, divs, quals, succs = [], [], [], [], [], [], []
        for _ in range(N_RUNS):
            seed = random.randint(0, 10000)  # Random seed cho variance
            res = run_algorithm(algo, dim=30, pop_size=100, seed=seed)
            fits.append(res['best_fit'])
            times.append(res['elapsed'])
            spaces.append(res['space'])
            convs.append(res['conv_speed'])
            #divs.append(res['diversity'])
            quals.append(res['quality'])
            succs.append(res['success'])
        
        # Tính mean/std
        robustness_data.append({
            'Algo': algo,
            'Mean Fitness': np.mean(fits), 'Std Fitness': np.std(fits),
            'Mean Time (s)': np.mean(times), 'Std Time': np.std(times),
            'Mean Space (bytes)': np.mean(spaces),
            'Mean Conv Speed (iters)': np.mean(convs),
            #'Mean Diversity': np.mean(divs),
            'Mean Solution Quality (norm)': np.mean(quals),
            'Success Rate (%)': np.mean(succs) * 100
        })
    
    pd.DataFrame(robustness_data).to_csv(f"results/robustness_{timestamp}.csv", index=False)
    print("Robustness CSV saved!")
    
    # Plot convergence curves (avg hist qua runs, cho convergence speed)
    # Giả sử collect avg_hist cho mỗi algo (add in loop nếu cần)
    plt.figure()
    # Example: for algo, plot np.mean(hists, axis=0) vs iterations
    plt.savefig(f"results/convergence_plot_{timestamp}.png")
    
    # 2. Scalability: Vary dim và pop_size
    scalability_data = []
    print("\n--- Measuring Scalability (Vary DIM and POP) ---")
    for dim in DIMS:
        for pop in POP_SIZES:
            for algo in ALGOS:
                res = run_algorithm(algo, dim=dim, pop_size=pop, seed=42)  # Fix seed cho fair
                scalability_data.append({
                    'Algo': algo, 'DIM': dim, 'POP': pop,
                    'Fitness': res['best_fit'], 'Time (s)': res['elapsed'], 'Space (bytes)': res['space']
                })
    
    df_scal = pd.DataFrame(scalability_data)
    df_scal.to_csv(f"results/scalability_{timestamp}.csv", index=False)
    print("Scalability CSV saved!")
    
    # Plot scalability: Bar chart time vs dim cho từng algo
    sns.barplot(data=df_scal, x='DIM', y='Time (s)', hue='Algo')
    plt.savefig(f"results/scalability_time_plot_{timestamp}.png")
    
    
    print("\nAll experiments done! Use CSVs/plots for report tables/charts.")

if __name__ == "__main__":
    run_experiments()