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
        print(f"Best Fitness: {gbest_value}")
        print(f"Best Solution: {gbest}")

        return gbest, gbest_value
