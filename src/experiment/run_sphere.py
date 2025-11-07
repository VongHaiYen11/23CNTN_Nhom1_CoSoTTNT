import numpy as np
import time
import os
import pandas as pd

# ==== Import các thuật toán ====
from src.algorithms.swarm_algorithms.FA import firefly_optimize
from src.algorithms.swarm_algorithms.ABC import abc_optimize
from src.algorithms.swarm_algorithms.Cuckoo import CuckooSearch
from src.algorithms.swarm_algorithms.PSO import pso_optimize
from src.algorithms.traditional_algorithms.GA import genetic_algorithm_optimize
from src.algorithms.traditional_algorithms.HC import hill_climbing_optimize
from src.algorithms.traditional_algorithms.SA import simulated_annealing_optimize

# ==== Import bài toán ====
from src.problem.continuous.sphere import sphere

# ==== Cấu hình ====
POP_SIZE = 200
MAX_ITERATIONS = 200
SEED = 42
DIM = 30
LOWER_BOUND = -5.12
UPPER_BOUND = 5.12


def run_sphere():
    print("\n--- CHẠY THÍ NGHIỆM - SPHERE FUNCTION ---")
    print(f"Pop Size = {POP_SIZE}, Max Iterations = {MAX_ITERATIONS}\n")

    # === Thiết lập các thuật toán cần chạy ===
    algorithms_to_run = {
        "FA": firefly_optimize,
        "ABC": abc_optimize
    }

    results = {}

    # # === 2 Chạy từng thuật toán ===
    # for name, algo_func in algorithms_to_run.items():
    #     print(f"--- Đang chạy {name} ---")

    #     start_time = time.time()

    #     # Gọi hàm tối ưu tương ứng (mỗi hàm tự có cách nhận tham số riêng)
    #     best_sol, best_fit, hist = algo_func(
    #         objective_function=sphere,   # tên tham số này tùy theo file thuật toán
    #         lower_bound=LOWER_BOUND,
    #         upper_bound=UPPER_BOUND,
    #         dimension=DIM,
    #         population_size=POP_SIZE,
    #         max_iterations=MAX_ITERATIONS,
    #         seed=SEED
    #     )

    #     elapsed = time.time() - start_time

    #     results[name] = {
    #         "Thuật toán": name,
    #         "Best Fitness": best_fit,
    #         "Thời gian (s)": elapsed
    #     }

    #     print(f"Kết quả {name}: Fitness = {best_fit:.6f}, Thời gian = {elapsed:.3f}s")

    #===== Firefly Algorithm =====
    # print("\n--- Đang chạy FA ---")
    # start = time.time()
    # fa_sol, fa_fit, fa_hist = firefly_optimize(
    #     objective_function=sphere,
    #     lower_bound=LOWER_BOUND,
    #     upper_bound=UPPER_BOUND,
    #     dimension=DIM,
    #     population_size=POP_SIZE,
    #     max_iterations=MAX_ITERATIONS,
    #     seed=SEED
    # )
    # results["FA"] = {"Thuật toán": "FA", "Best Fitness": fa_fit, "Thời gian (s)": time.time() - start}

    # ===== Artificial Bee Colony =====
    print("\n--- Đang chạy ABC ---")
    start = time.time()
    abc_sol, abc_fit, abc_hist = abc_optimize(
        fitness_function=sphere,     
        lower_bound=LOWER_BOUND,
        upper_bound=UPPER_BOUND,
        problem_size=DIM,
        num_employed_bees=POP_SIZE,
        num_onlooker_bees=POP_SIZE,
        max_iterations=MAX_ITERATIONS,
        limit=50,
        seed=SEED
    )
    results["ABC"] = {"Thuật toán": "ABC", "Best Fitness": abc_fit, "Thời gian (s)": time.time() - start}

    # ===== Cuckoo =====
    print("\n--- Đang chạy Cuckoo ---")
    start = time.time()
    best_sol, Cuckoo_fit = CuckooSearch(
        fitness_func=sphere,
        lower_bound=LOWER_BOUND,
        upper_bound=UPPER_BOUND,
        dim=DIM,
        population_size=POP_SIZE,
        max_iter=1000,
        seed=SEED
    ).run()
    results["Cuckoo"] = {"Thuật toán": "Cuckoo", "Best Fitness": Cuckoo_fit, "Thời gian (s)": time.time() - start}

    # ===== Particle Swarm Optimization =====
    print("\n--- Đang chạy PSO ---")
    start = time.time()
    best_pos, best_val, hist = pso_optimize(
    objective_function=sphere,
    lower_bound=LOWER_BOUND,
    upper_bound=UPPER_BOUND,
    dimension=DIM,
    population_size=POP_SIZE,
    max_iterations=MAX_ITERATIONS,
    seed=SEED
    )
    results["PSO"] = {"Thuật toán": "PSO", "Best Fitness": best_val, "Thời gian (s)": time.time() - start}

    # ==== Hill climbing ====
    print("\n--- Đang chạy HC ---")
    start = time.time()
    best_hc, best_val = hill_climbing_optimize(
    fitness_func=sphere,
    x_min=LOWER_BOUND,
    x_max=UPPER_BOUND,
    dimension=DIM,
    max_iteration=MAX_ITERATIONS,
    seed=SEED
    )
    results["HC"] = {"Thuật toán": "HC", "Best Fitness": best_val, "Thời gian (s)": time.time() - start}

    # ==== Genetic Algorithm ====
    print("\n--- Đang chạy GA ---")
    start = time.time()
    best_ga, best_val = genetic_algorithm_optimize(
    fitness_func=sphere,
    x_min=LOWER_BOUND,
    x_max=UPPER_BOUND,
    dimension=DIM,
    npopulation=POP_SIZE,
    max_iteration=MAX_ITERATIONS,
    seed=SEED
    )
    results["GA"] = {"Thuật toán": "GA", "Best Fitness": best_val, "Thời gian (s)": time.time() - start}

    # ==== Simulated Annealing ====
    print("\n--- Đang chạy SA ---")
    start = time.time()
    best_sa, best_val = simulated_annealing_optimize(
    fitness_func=sphere,
    x_min=LOWER_BOUND,
    x_max=UPPER_BOUND,
    dimension=DIM,
    max_iteration=70000,
    seed=SEED
    )
    results["SA"] = {"Thuật toán": "SA", "Best Fitness": best_val, "Thời gian (s)": time.time() - start}

    # # === Lưu file CSV ===
    # os.makedirs("results", exist_ok=True)
    # timestamp = time.strftime("%Y%m%d-%H%M%S")
    # csv_path = f"results/sphere_results_{timestamp}.csv"
    # pd.DataFrame(results.values()).to_csv(csv_path, index=False)
    # print(f"\n Kết quả đã lưu tại: {csv_path}")

    # === 4 In bảng tổng hợp ===
    print("\n TỔNG KẾT (SPHERE)")
    print(f"{'Thuật toán':<10} | {'Best Fitness':<15} | {'Thời gian (s)':<10}")
    print("-" * 45)
    for r in results.values():
        print(f"{r['Thuật toán']:<10} | {r['Best Fitness']:<15.6f} | {r['Thời gian (s)']:<10.3f}")


if __name__ == "__main__":
    run_sphere()
