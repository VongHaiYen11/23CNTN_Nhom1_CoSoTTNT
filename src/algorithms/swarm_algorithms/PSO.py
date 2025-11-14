import numpy as np

class ParticleSwarmOptimization:
    def __init__(self, objective_function, lower_bound, upper_bound, dim,
                 population_size=30, max_iter=1000, w=0.7, c1=1.5, c2=1.5,
                 seed=None):
        """
        Particle Swarm Optimization (PSO) - General version for continuous problems.
        """
        if seed is not None:
            np.random.seed(seed)

        self.fitness_func = objective_function
        self.lower_bound = np.array(lower_bound)
        self.upper_bound = np.array(upper_bound)
        self.dim = dim
        self.population_size = population_size
        self.max_iter = max_iter
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.seed = seed

    def simple_bounds(self, x):
        """Giữ cá thể trong giới hạn."""
        return np.clip(x, self.lower_bound, self.upper_bound)

    def update_personal_best(self, positions, pbest, pbest_values):
        """Cập nhật best cá nhân cho từng hạt."""
        for i in range(self.population_size):
            fitness = self.fitness_func(positions[i])
            if fitness < pbest_values[i]:
                pbest[i] = positions[i]
                pbest_values[i] = fitness
        return pbest, pbest_values

    def update_global_best(self, pbest, pbest_values, gbest, gbest_value):
        """Cập nhật best toàn cục."""
        best_idx = np.argmin(pbest_values)
        if pbest_values[best_idx] < gbest_value:
            gbest = pbest[best_idx].copy()
            gbest_value = pbest_values[best_idx]
        return gbest, gbest_value

    def update_velocity(self, velocities, positions, pbest, gbest):
        """Cập nhật vận tốc theo công thức PSO."""
        r1 = np.random.rand(self.population_size, self.dim)
        r2 = np.random.rand(self.population_size, self.dim)
        new_velocities = (
            self.w * velocities
            + self.c1 * r1 * (pbest - positions)
            + self.c2 * r2 * (gbest - positions)
        )
        return new_velocities

    def update_position(self, positions, velocities):
        """Cập nhật vị trí hạt và giới hạn trong bound."""
        new_positions = positions + velocities
        return self.simple_bounds(new_positions)

    def run(self, verbose=False):
        """Thực thi thuật toán PSO."""
        rng = np.random.default_rng(self.seed)

        # --- Khởi tạo ---
        positions = rng.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        velocities = np.zeros((self.population_size, self.dim))
        pbest = positions.copy()
        pbest_values = np.array([self.fitness_func(x) for x in positions])
        gbest = positions[np.argmin(pbest_values)]
        gbest_value = np.min(pbest_values)
        history = [gbest_value]

        # --- Vòng lặp chính ---
        for t in range(self.max_iter):
            pbest, pbest_values = self.update_personal_best(positions, pbest, pbest_values)
            gbest, gbest_value = self.update_global_best(pbest, pbest_values, gbest, gbest_value)
            velocities = self.update_velocity(velocities, positions, pbest, gbest)
            positions = self.update_position(positions, velocities)
            history.append(gbest_value)

            if verbose and (t % 50 == 0 or t == self.max_iter - 1):
                print(f"Iteration {t+1}/{self.max_iter}: best fitness = {gbest_value:.6f}")

        print("\n--- Optimization Results (PSO) ---")
        # print(f"Best Fitness: {gbest_value}")
        # print(f"Best Solution: {gbest}")

        return gbest, gbest_value, history


import numpy as np

class ParitcleSwarmKnapsack:
    def __init__(self, weights, values, capacity, dim=None,
                 population_size=30, max_iter=1000, w=0.7, c1=1.5, c2=1.5,
                 seed=None, verbose=True):
        """
        Particle Swarm Optimization (PSO) cho bài toán Knapsack (rời rạc 0-1)
        """
        if seed is not None:
            np.random.seed(seed)

        self.weights = np.array(weights)
        self.values = np.array(values)
        self.capacity = capacity
        self.dim = dim if dim is not None else len(weights)

        self.population_size = population_size
        self.max_iter = max_iter
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.verbose = verbose
        self.seed = seed

    def fitness(self, solution):
        """Tính giá trị tổng (chỉ hợp lệ mới tính)."""
        total_weight = np.sum(solution * self.weights)
        total_value = np.sum(solution * self.values)
        if total_weight > self.capacity:
            return 0  # hoặc -inf, nhưng 0 giúp tránh lỗi nan
        return total_value

    def sigmoid_solution(self, x):
        """Ánh xạ giá trị thực sang nhị phân và đảm bảo hợp lệ."""
        s = 1 / (1 + np.exp(-x))
        sol = (s > 0.5).astype(int)
        while np.sum(sol * self.weights) > self.capacity:
            ones = np.where(sol == 1)[0]
            if len(ones) == 0:
                break
            sol[np.random.choice(ones)] = 0
        return sol

    def initialize_population(self):
        """Khởi tạo quần thể ngẫu nhiên trong [-4, 4]."""
        positions = np.random.uniform(-4, 4, (self.population_size, self.dim))
        velocities = np.zeros((self.population_size, self.dim))
        return positions, velocities

    def run(self):
        """Main PSO loop."""
        positions, velocities = self.initialize_population()

        # Chuyển sang nhị phân và tính fitness
        binary_positions = np.array([self.sigmoid_solution(x) for x in positions])
        fitness = np.array([self.fitness(x) for x in binary_positions])

        pbest = positions.copy()
        pbest_fitness = fitness.copy()
        gbest_idx = np.argmax(fitness)
        gbest = positions[gbest_idx].copy()
        gbest_fitness = fitness[gbest_idx]

        history = []
        for t in range(1, self.max_iter + 1):
            r1 = np.random.rand(self.population_size, self.dim)
            r2 = np.random.rand(self.population_size, self.dim)

            velocities = (
                self.w * velocities
                + self.c1 * r1 * (pbest - positions)
                + self.c2 * r2 * (gbest - positions)
            )
            positions = positions + velocities

            # Cập nhật cá thể nhị phân và fitness
            binary_positions = np.array([self.sigmoid_solution(x) for x in positions])
            fitness = np.array([self.fitness(x) for x in binary_positions])

            # Cập nhật pbest
            improved = fitness > pbest_fitness
            pbest[improved] = positions[improved]
            pbest_fitness[improved] = fitness[improved]

            # Cập nhật gbest
            best_idx = np.argmax(fitness)
            if fitness[best_idx] > gbest_fitness:
                gbest = positions[best_idx].copy()
                gbest_fitness = fitness[best_idx]

            history.append(gbest_fitness)
            if self.verbose and (t % 50 == 0 or t == self.max_iter):
                print(f"Iteration {t}/{self.max_iter}: best fitness = {gbest_fitness}")

        # Giải pháp nhị phân cuối cùng
        best_solution = self.sigmoid_solution(gbest)
        best_value = self.fitness(best_solution)

        print("\n=== Final Result (PSO-Knapsack) ===")
        print(f"Best fitness (total value): {best_value}")
        print(f"Best solution: {best_solution}")
        print(f"Total weight: {np.sum(best_solution * self.weights)}")

        return best_solution, best_value, history