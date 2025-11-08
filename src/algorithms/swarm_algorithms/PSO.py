# import numpy as np

# def pso_optimize(
#     objective_function,
#     dimension,
#     lower_bound,
#     upper_bound,
#     population_size,
#     max_iterations,
#     seed,
#     w=0.7,
#     c1=1.5,
#     c2=1.5
# ):
#     """
#     Particle Swarm Optimization (PSO) tổng quát cho các bài toán liên tục.

#     Parameters
#     ----------
#     objective_function : callable
#         Hàm mục tiêu cần tối ưu hóa (minimization).
#     dim : int
#         Số chiều của không gian tìm kiếm.
#     bounds : tuple
#         (min, max) cho các giá trị biến.
#     n_particles : int
#         Số lượng hạt trong đàn.
#     n_iterations : int
#         Số vòng lặp tối đa.
#     w, c1, c2 : float
#         Tham số quán tính và hệ số ảnh hưởng cá nhân / bầy đàn.
#     seed : int
#         Giá trị ngẫu nhiên để tái lập kết quả.

#     Returns
#     -------
#     gbest : np.ndarray
#         Vị trí tốt nhất tìm được.
#     gbest_value : float
#         Giá trị fitness tốt nhất.
#     history : list
#         Lịch sử giá trị tốt nhất qua các vòng lặp.
#     """

#     rng = np.random.default_rng(seed)

#     # --- Khởi tạo ---
#     positions = rng.uniform(lower_bound, upper_bound, (population_size, dimension))
#     velocities = np.zeros((population_size, dimension))
#     pbest = positions.copy()
#     pbest_values = np.array([objective_function(x) for x in positions])
#     gbest = positions[np.argmin(pbest_values)]
#     gbest_value = np.min(pbest_values)
#     history = [gbest_value]

#     # --- Vòng lặp chính ---
#     for t in range(max_iterations):
#         for i in range(population_size):
#             fitness = objective_function(positions[i])

#             # Cập nhật best cá nhân
#             if fitness < pbest_values[i]:
#                 pbest[i] = positions[i]
#                 pbest_values[i] = fitness

#         # Cập nhật best toàn cục
#         best_idx = np.argmin(pbest_values)
#         if pbest_values[best_idx] < gbest_value:
#             gbest = pbest[best_idx].copy()
#             gbest_value = pbest_values[best_idx]

#         # Cập nhật vận tốc và vị trí
#         r1, r2 = rng.random((population_size, dimension)), rng.random((population_size, dimension))
#         velocities = (
#             w * velocities
#             + c1 * r1 * (pbest - positions)
#             + c2 * r2 * (gbest - positions)
#         )
#         positions += velocities

#         # Giữ trong giới hạn
#         positions = np.clip(positions, lower_bound, upper_bound)
#         history.append(gbest_value)

#     return gbest, gbest_value, history

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

        return gbest, gbest_value
