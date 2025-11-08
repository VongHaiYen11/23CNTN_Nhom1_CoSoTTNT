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
            if verbose and (it % 20 == 0 or it == self.max_iter - 1):
                print(f"Iter {it+1}/{self.max_iter}: best fitness = {self.best_fitness:.6f}")

        return self.best_solution, self.best_fitness

