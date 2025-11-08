import numpy as np
import time
import os
import pandas as pd

# ==== Import các thuật toán ====
from src.algorithms.swarm_algorithms.ABC import abc_optimize
from src.algorithms.swarm_algorithms.FA import firefly_optimize  
from src.algorithms.swarm_algorithms.ACO import AntColonyOptimizationDiscrete
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
        # "ABC": abc_optimize,
        "ACO": AntColonyOptimizationDiscrete
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

        # ---- THÊM KHỐI "ACO" MỚI ----
        elif name == "ACO":
            # Class ACO_Discrete của chúng ta đã xử lý nghiệm rời rạc (0/1)
            # Ta chỉ cần cung cấp hàm fitness (minimize)
            def fitness(solution): # solution đã là [0, 1]
                val = knapsack_fitness(solution, weights, values, capacity)
                # Chuyển sang bài toán minimize (1 / value)
                return 1.0 / (val + 1e-8)

            # Khởi tạo class (algo_func ở đây là ACO_Discrete)
            best_sol, best_fit, hist = AntColonyOptimizationDiscrete(
                fitness_func=fitness,
                problem_size=N_ITEMS,
                n_ants=POP_SIZE,
                n_iterations=MAX_ITERATIONS,
                alpha=1.0,
                rho=0.5,
                Q=1.0,
                seed=SEED
            ).run()
            best_value = 1.0 / best_fit # Đảo ngược lại

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
