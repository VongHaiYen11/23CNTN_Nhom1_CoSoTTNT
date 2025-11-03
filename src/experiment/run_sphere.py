import numpy as np
import time
import os
import pandas as pd

# ==== Import các thuật toán ====
from src.algorithms.FA import firefly_optimize
from src.algorithms.ABC import abc_optimize
from src.algorithms.Cuckoo import cuckoo_optimize
from src.algorithms.PSO import pso_optimize

# ==== Import bài toán ====
from src.problem.continuous.sphere import sphere

# ==== Cấu hình ====
POP_SIZE = 100
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
    best_sol, Cuckoo_fit = cuckoo_optimize(
    fitness_func=sphere,
    xmin=-5.12,
    xmax=5.12,
    dimension=DIM,
    population_size=POP_SIZE,
    max_iterations=MAX_ITERATIONS,
    seed=42
    )
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
