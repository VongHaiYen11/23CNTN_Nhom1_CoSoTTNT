import numpy as np

class HillClimbing:
    def __init__(self, fitness_func, lower_bound, upper_bound, dim=1,
                 max_iter=100, n_neighbors=1000, seed=None, verbose=False):
        if seed is not None:
            np.random.seed(seed)

        self.fitness_func = fitness_func
        self.lower_bound = np.array(lower_bound)
        self.upper_bound = np.array(upper_bound)
        self.dim = dim
        self.max_iter = max_iter
        self.n_neighbors = n_neighbors
        self.verbose = verbose

    def run(self):
        # Khởi tạo nghiệm ban đầu ngẫu nhiên
        current_solution = np.random.uniform(self.lower_bound, self.upper_bound, size=self.dim)
        current_fitness = self.fitness_func(current_solution)
        hist = [current_fitness]

        step = (self.upper_bound - self.lower_bound) * 0.1

        for iteration in range(self.max_iter):
            # Sinh tất cả neighbor (vector hóa)
            perturbations = np.random.uniform(-step, step, size=(self.n_neighbors, self.dim))
            neighbors = np.clip(current_solution + perturbations, self.lower_bound, self.upper_bound)

            # Đánh giá toàn bộ neighbor
            fitness_values = np.apply_along_axis(self.fitness_func, 1, neighbors)
            best_idx = np.argmin(fitness_values)
            best_neighbor = neighbors[best_idx]
            best_neighbor_fitness = fitness_values[best_idx]

            # Cập nhật nếu có cải thiện
            if best_neighbor_fitness < current_fitness:
                current_solution = best_neighbor
                current_fitness = best_neighbor_fitness
            else:
                if self.verbose:
                    print("No improvement, stopping.")
                break  # Không cải thiện → hội tụ
            hist.append(current_fitness)
            
            # if self.verbose:
            #     print(f"Iteration {iteration + 1}: best fitness = {current_fitness:.6f} \nbest_solution = {current_solution}")
        
        print("\n--- Optimization Results (Hill Climbing) ---")
        return current_solution, current_fitness, hist

class HillClimbingKnapsack:
    def __init__(self, weights, values, capacity, dim=None,
                 max_iter=100, n_neighbors=10, seed=None, verbose=False):
        if seed is not None:
            np.random.seed(seed)

        self.weights = np.array(weights)
        self.values = np.array(values)
        self.capacity = capacity
        self.dim = dim if dim is not None else len(weights)

        self.max_iter = max_iter
        self.n_neighbors = n_neighbors
        self.verbose = verbose

    def fitness(self, solution):
        """Tính tổng giá trị solution."""
        return np.sum(solution * self.values)

    def validate(self, solution):
        """Kiểm tra tổng trọng lượng ≤ capacity."""
        return np.sum(solution * self.weights) <= self.capacity

    def generate_neighbors(self, solution):
        """Sinh neighbor hợp lệ bằng cách flip 1 bit ngẫu nhiên."""
        neighbors = []
        for _ in range(self.n_neighbors):
            while True:
                neighbor = solution.copy()
                i = np.random.randint(0, self.dim)
                neighbor[i] = 1 - neighbor[i]
                if self.validate(neighbor):
                    neighbors.append(neighbor)
                    break
        return neighbors

    def run(self):
        # Khởi tạo giải hợp lệ
        best_solution = np.random.randint(0, 2, size=self.dim)
        while not self.validate(best_solution):
            best_solution = np.random.randint(0, 2, size=self.dim)
        best_fitness = self.fitness(best_solution)

        hist = [best_fitness]
        for iteration in range(1, self.max_iter + 1):
            neighbors = self.generate_neighbors(best_solution)
            if not neighbors:
                break

            # Chọn neighbor tốt nhất
            fitness_values = np.array([self.fitness(n) for n in neighbors])
            idx_best = np.argmax(fitness_values)
            best_neighbor = neighbors[idx_best]
            best_neighbor_fitness = fitness_values[idx_best]

            # Cập nhật nếu tốt hơn
            if best_neighbor_fitness > best_fitness:
                best_solution = best_neighbor
                best_fitness = best_neighbor_fitness
            else:
                break  # không cải thiện → hội tụ
            
            hist.append(best_fitness)
            if self.verbose:
                print(f"Iteration {iteration}: best fitness = {best_fitness:.6f}")

        return best_solution, best_fitness, hist