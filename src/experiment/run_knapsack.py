import numpy as np
import time
import os
import pandas as pd

# ==== Import các thuật toán ====
from src.algorithms.ABC import abc_optimize
from src.algorithms.FA import firefly_optimize  
# ==== Import bài toán ====
from src.problem.discrete.knapsack import knapsack_fitness, generate_knapsack_problem

# ==== Cấu hình ====
POP_SIZE = 30
MAX_ITERATIONS = 100
SEED = 42
N_ITEMS = 20  # số vật phẩm trong bài toán


def run_knapsack():
    print("\n--- CHẠY THÍ NGHIỆM - KNAPSACK PROBLEM ---")
    print(f"Pop Size = {POP_SIZE}, Max Iterations = {MAX_ITERATIONS}, Items = {N_ITEMS}\n")

    # === 1 Sinh dữ liệu bài toán ===
    weights, values, capacity = generate_knapsack_problem(n_items=N_ITEMS, seed=SEED)

    print(f"Sức chứa (capacity): {capacity}")
    print(f"Trọng lượng: {weights}")
    print(f"Giá trị: {values}\n")

    # === 2 Thiết lập thuật toán cần chạy ===
    algorithms_to_run = {
        "ABC": abc_optimize,
    }

    results = {}

    # === 3 Chạy từng thuật toán ===
    for name, algo_func in algorithms_to_run.items():
        print(f"\n--- Đang chạy {name} ---")
        start = time.time()

        if name == "ABC":
            # ABC có thể áp dụng cho không gian rời rạc nếu ta biểu diễn binary solution → [0,1]
            def fitness(solution):
                # Giải mã nghiệm nhị phân sang giá trị fitness (dùng dấu âm để minimization)
                return -knapsack_fitness(solution, weights, values, capacity)

            best_sol, best_fit, hist = abc_optimize(
                fitness_function=fitness,
                lower_bound=0,
                upper_bound=1,
                problem_size=N_ITEMS,
                num_employed_bees=POP_SIZE,
                num_onlooker_bees=POP_SIZE,
                max_iterations=MAX_ITERATIONS,
                limit=50,
                seed=SEED
            )
            best_value = -best_fit  # đổi lại về giá trị dương vì ta đã đảo dấu

        elif name == "FA":
            # Nếu có phiên bản FA rời rạc (ví dụ dùng round)
            def fitness(solution):
                return -knapsack_fitness(np.round(solution), weights, values, capacity)

            best_sol, best_fit, hist = firefly_optimize(
                objective_function=fitness,
                lower_bound=0,
                upper_bound=1,
                dimension=N_ITEMS,
                population_size=POP_SIZE,
                max_iterations=MAX_ITERATIONS,
                seed=SEED
            )
            best_value = -best_fit

        elapsed = time.time() - start

        results[name] = {
            "Thuật toán": name,
            "Best Fitness": best_value,
            "Thời gian (s)": elapsed
        }

        print(f"Kết quả {name}: Tổng giá trị = {best_value:.2f}, Thời gian = {elapsed:.3f}s")

    # === 4 Lưu file CSV ===
    # os.makedirs("results", exist_ok=True)
    # timestamp = time.strftime("%Y%m%d-%H%M%S")
    # csv_path = f"results/knapsack_results_{timestamp}.csv"
    # pd.DataFrame(results.values()).to_csv(csv_path, index=False)
    # print(f"\n Kết quả đã lưu tại: {csv_path}")

    # === 5 In bảng tổng hợp ===
    print("\n TỔNG KẾT (KNAPSACK)")
    print(f"{'Thuật toán':<10} | {'Tổng giá trị':<15} | {'Thời gian (s)':<10}")
    print("-" * 45)
    for r in results.values():
        print(f"{r['Thuật toán']:<10} | {r['Best Fitness']:<15.2f} | {r['Thời gian (s)']:<10.3f}")


if __name__ == "__main__":
    run_knapsack()
