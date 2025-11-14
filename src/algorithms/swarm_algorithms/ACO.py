# Continuous problem using ACOr and archive 
import numpy as np
import math

class AntColonyOptimizationContinuous:
    """
    Tối ưu Đàn kiến (ACOr) cho các bài toán Liên tục (ví dụ: Sphere).
    Sử dụng cơ chế "Kho lời giải" (Archive).
    """
    def __init__(self, objective_function, dim,
                 lower_bound, upper_bound,
                 population_size=30, max_iter=100,
                 archive_size=None, # Kích thước kho, thường bằng population_size
                 rho=0.85,          # Tốc độ học / Tốc độ hội tụ
                 seed=None):
        
        self.fitness_func = objective_function
        self.dim = dim
        self.population_size = population_size
        self.max_iter = max_iter
        self.rng = np.random.default_rng(seed)

        self.lower_bound = np.array(lower_bound)
        self.upper_bound = np.array(upper_bound)
        self.rho = rho # Trong ACOr, rho giống như tốc độ học
        
        if archive_size is None:
            self.archive_size = self.population_size
        else:
            self.archive_size = archive_size
            
        # Khởi tạo kho lưu trữ
        self.archive = self.rng.uniform(self.lower_bound, self.upper_bound,
                                         (self.archive_size, self.dim))
        self.archive_fitness = np.array([self.fitness_func(s) for s in self.archive])
        self._sort_archive()
        
        # Tính toán trọng số (weights) cho việc chọn từ kho lưu trữ
        self.archive_weights = self._calculate_acor_weights()
        
        self.best_solution = self.archive[0].copy()
        self.best_fitness = self.archive_fitness[0]
        self.history = []

    def _sort_archive(self):
        """Sắp xếp kho lưu trữ theo fitness (tốt nhất lên đầu)."""
        sort_indices = np.argsort(self.archive_fitness)
        self.archive = self.archive[sort_indices]
        self.archive_fitness = self.archive_fitness[sort_indices]

    def _calculate_acor_weights(self):
        """Tính trọng số (pheromone) cho các giải pháp trong kho."""
        k = self.archive_size
        ranks = np.arange(k)
        q = 0.05 # Tham số q
        weights = (1 / (q * k * np.sqrt(2 * np.pi))) * \
                  np.exp(-ranks**2 / (2 * q**2 * k**2))
        return weights / np.sum(weights)

    def _construct_solutions(self):
        """Tạo lời giải mới bằng cách lấy mẫu xung quanh kho lưu trữ."""
        all_solutions = np.zeros((self.population_size, self.dim))
        for k in range(self.population_size):
            # 1. Chọn một lời giải "hướng dẫn" từ kho (dựa trên trọng số)
            guide_idx = self.rng.choice(self.archive_size, p=self.archive_weights)
            guide_sol = self.archive[guide_idx]

            # 2. Tạo lời giải mới bằng cách lấy mẫu Gaussian quanh guide_sol
            new_sol = np.zeros(self.dim)
            for d in range(self.dim):
                # Tính độ lệch chuẩn (sigma) cho chiều d
                sigma_d = self.rho * np.mean(np.abs(self.archive[:, d] - guide_sol[d]))
                
                # Lấy mẫu
                new_sol[d] = self.rng.normal(loc=guide_sol[d], scale=sigma_d)
            
            # 3. Giữ lời giải trong biên
            all_solutions[k] = np.clip(new_sol, self.lower_bound, self.upper_bound)
        return all_solutions

    def _update_archive(self, new_solutions, new_fitness):
        """Cập nhật kho lưu trữ: gộp và giữ lại K lời giải tốt nhất."""
        combined_solutions = np.vstack((self.archive, new_solutions))
        combined_fitness = np.concatenate((self.archive_fitness, new_fitness))

        sort_indices = np.argsort(combined_fitness)
        self.archive = combined_solutions[sort_indices][:self.archive_size]
        self.archive_fitness = combined_fitness[sort_indices][:self.archive_size]

    def run(self, verbose=False):
        """Vòng lặp chính cho bài toán liên tục (ACOr)."""
        self.best_solution = self.archive[0].copy()
        self.best_fitness = self.archive_fitness[0]
        self.history = []

        for it in range(self.max_iter):
            # 1. Kiến tạo lời giải (lấy mẫu từ kho)
            all_solutions = self._construct_solutions()
            all_fitness = np.array([self.fitness_func(s) for s in all_solutions])

            # 2. Cập nhật kho lưu trữ
            self._update_archive(all_solutions, all_fitness)

            # 3. Cập nhật lời giải tốt nhất toàn cục
            if self.archive_fitness[0] < self.best_fitness:
                self.best_fitness = self.archive_fitness[0]
                self.best_solution = self.archive[0].copy()

            self.history.append(self.best_fitness)

            
            # if verbose and (it % 20 == 0 or it == self.max_iter - 1):
            #     print(f"Iter {it+1}/{self.max_iter}: best fitness = {self.best_fitness:.6f}")
        print("\n--- Optimization Results (ACOr) --- ")
        return self.best_solution, self.best_fitness, self.history


import numpy as np

import numpy as np

class AntColonyOptimizationKnapsack:
    """
    Ant Colony Optimization (ACO) cho bài toán Knapsack (0-1).
    Tối đa hóa tổng giá trị, đảm bảo không vượt quá sức chứa.
    """

    def __init__(self, weights, values, capacity,
                 n_ants=30, max_iter=100,
                 alpha=1.0, beta=2.0, rho=0.3, Q=1.0,
                 seed=None, verbose=True):
        self.weights = np.array(weights)
        self.values = np.array(values)
        self.capacity = capacity
        self.n_items = len(weights)

        self.n_ants = n_ants
        self.max_iter = max_iter
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.Q = Q
        self.verbose = verbose

        self.rng = np.random.default_rng(seed)

        # --- Khởi tạo pheromone và heuristic ---
        self.tau = np.ones(self.n_items)              # Pheromone ban đầu
        self.mu = self.values / (self.weights + 1e-9) # heuristic: value/weight

        self.best_solution = None
        self.best_fitness = 0
        self.history = []

    # =========================================================
    # ====== HÀM ĐÁNH GIÁ =====================================
    def fitness(self, solution):
        """Tính tổng giá trị của lời giải nếu hợp lệ, ngược lại trả 0."""
        total_weight = np.sum(solution * self.weights)
        total_value = np.sum(solution * self.values)
        if total_weight > self.capacity:
            return 0
        return total_value

    # =========================================================
    # ====== TÍNH XÁC SUẤT CHỌN ITEM ==========================
    def _probabilities(self, allowed):
        """Tính xác suất chọn item trong allowed theo công thức p_j."""
        tau_allowed = self.tau[allowed] ** self.alpha
        mu_allowed = self.mu[allowed] ** self.beta
        numerator = tau_allowed * mu_allowed
        denom = np.sum(numerator)
        if denom == 0:
            return np.ones_like(numerator) / len(numerator)
        return numerator / denom

    # =========================================================
    # ====== XÂY DỰNG LỜI GIẢI ================================
    def _construct_solution(self):
        """Tạo lời giải hợp lệ cho một con kiến."""
        solution = np.zeros(self.n_items, dtype=int)
        current_weight = 0
        available = np.arange(self.n_items)

        while len(available) > 0:
            allowed = available[self.weights[available] + current_weight <= self.capacity]
            if len(allowed) == 0:
                break

            probs = self._probabilities(allowed)
            chosen = self.rng.choice(allowed, p=probs)

            solution[chosen] = 1
            current_weight += self.weights[chosen]
            available = available[available != chosen]

        fitness = self.fitness(solution)
        return solution, fitness

    # =========================================================
    # ====== CẬP NHẬT PHEROMONE ===============================
    def _update_pheromone(self, all_solutions, all_fitness):
        """Cập nhật pheromone sau mỗi vòng."""
        self.tau *= (1 - self.rho)  # bay hơi

        # Cập nhật pheromone dựa trên fitness (tối đa hóa)
        for sol, fit in zip(all_solutions, all_fitness):
            if fit > 0:
                delta_tau = self.Q * fit
                self.tau += delta_tau * sol

    # =========================================================
    # ====== VÒNG LẶP CHÍNH ==================================
    def run(self):
        for it in range(1, self.max_iter + 1):
            all_solutions = []
            all_fitness = []

            for _ in range(self.n_ants):
                sol, fit = self._construct_solution()
                all_solutions.append(sol)
                all_fitness.append(fit)

            all_fitness = np.array(all_fitness)
            best_idx = np.argmax(all_fitness)

            if all_fitness[best_idx] > self.best_fitness:
                self.best_fitness = all_fitness[best_idx]
                self.best_solution = all_solutions[best_idx].copy()

            self.history.append(self.best_fitness)

            self._update_pheromone(all_solutions, all_fitness)

            if self.verbose and (it % 10 == 0 or it == self.max_iter):
                print(f"Iter {it}/{self.max_iter}: best fitness = {self.best_fitness:.4f}")

        print("\n=== Final Result (ACO-Knapsack) ===")
        print(f"Best solution: {self.best_solution}")
        print(f"Total value: {self.best_fitness}")
        print(f"Total weight: {np.sum(self.best_solution * self.weights)}")

        return self.best_solution, self.best_fitness, self.history