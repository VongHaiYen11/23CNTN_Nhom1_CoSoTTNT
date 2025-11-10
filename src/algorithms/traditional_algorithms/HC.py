import numpy as np

class HillClimbing:
  def __init__(self, fitness_func, lower_bound, upper_bound, dim=1,
               max_iter=100, n_neighbors=1000, tolerance=1e-4,
               seed=None, verbose=False):
    if seed is not None:
      np.random.seed(seed)

    self.fitness_func = fitness_func
    self.lower_bound = np.array(lower_bound)
    self.upper_bound = np.array(upper_bound)
    self.dim = dim
    self.max_iter = max_iter
    self.n_neighbors = n_neighbors
    self.tolerance = tolerance
    self.verbose = verbose

    # Khởi tạo nghiệm ban đầu
    self.current_solution = np.zeros(self.dim)
    self.current_fitness = self.fitness_func(self.current_solution)

  def run(self):
    step = (self.upper_bound - self.lower_bound) * 0.1

    for iteration in range(self.max_iter):
      # Sinh tất cả neighbor (vector hóa)
      perturbations = np.random.uniform(-step, step, size=(self.n_neighbors, self.dim))
      neighbors = np.clip(self.current_solution + perturbations, self.lower_bound, self.upper_bound)

      # Đánh giá toàn bộ neighbor
      fitness_values = np.apply_along_axis(self.fitness_func, 1, neighbors)
      best_idx = np.argmin(fitness_values)
      best_neighbor = neighbors[best_idx]
      best_neighbor_fitness = fitness_values[best_idx]

      # Cập nhật nếu có cải thiện
      if best_neighbor_fitness < self.current_fitness:
        print("Cap nhat", iteration)
        self.current_solution = best_neighbor
        self.current_fitness = best_neighbor_fitness
      else:
        print("Dung cap nhat", iteration)
        print(self.current_solution, self.current_fitness)
        break  # Không cải thiện → hội tụ

      if self.verbose:
        print(f"Iteration {iteration + 1}: best fitness = {self.current_fitness:.6f} \n, best_solution = {self.current_solution}")

    return self.current_solution, self.current_fitness
