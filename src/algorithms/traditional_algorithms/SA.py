import numpy as np

import numpy as np

class SimulatedAnnealing:
  def __init__(self, fitness_func, lower_bound, upper_bound, dim=1,
               max_iter=100, step_size=0.1, initial_temp=100,
               seed=None, verbose=False):
    if seed is not None:
      np.random.seed(seed)

    self.fitness_func = fitness_func
    self.lower_bound = np.array(lower_bound)
    self.upper_bound = np.array(upper_bound)
    self.dim = dim
    self.max_iter = max_iter
    self.step_size = step_size
    self.initial_temp = initial_temp
    self.verbose = verbose

  def run(self):
    current_solution = np.random.uniform(self.lower_bound, self.upper_bound, size=self.dim)
    current_fitness = self.fitness_func(current_solution)
    best_solution = current_solution.copy()
    best_fitness = current_fitness

    for i in range(1, self.max_iter + 1):
      temp = self.initial_temp * (1 - i / self.max_iter)
      if temp < 1e-8:
        temp = 1e-8

      perturbation = np.random.uniform(-self.step_size, self.step_size, size=self.dim)
      candidate = np.clip(current_solution + perturbation, self.lower_bound, self.upper_bound)
      candidate_fitness = self.fitness_func(candidate)

      accept = (
        candidate_fitness < current_fitness
        or np.random.rand() < np.exp(-(candidate_fitness - current_fitness) / temp)
      )

      if accept:
        current_solution = candidate
        current_fitness = candidate_fitness
        if current_fitness < best_fitness:
          best_solution = current_solution.copy()
          best_fitness = current_fitness

      if self.verbose:
        print(f"Iteration {i}: Temp={temp:.4f}, Best fitness={best_fitness:.6f}")

    return best_solution, best_fitness

class SimulatedAnnealingKnapsack:
    def __init__(self, weights, values, capacity, dim=None,
                 max_iter=100, step_size=0.1, initial_temp=100, seed=None, verbose=False):
        if seed is not None:
            np.random.seed(seed)

        self.weights = np.array(weights)
        self.values = np.array(values)
        self.capacity = capacity
        self.dim = dim if dim is not None else len(weights)

        self.max_iter = max_iter
        self.step_size = step_size
        self.initial_temp = initial_temp
        self.verbose = verbose

    def fitness(self, solution):
        """Tính tổng giá trị của solution."""
        return np.sum(solution * self.values)

    def validate(self, solution):
        """Kiểm tra tổng trọng lượng ≤ capacity."""
        return np.sum(solution * self.weights) <= self.capacity

    def generate_neighbor(self, solution):
        """Sinh neighbor hợp lệ bằng cách flip ngẫu nhiên các bit."""
        while True:
            neighbor = solution.copy()
            for i in range(self.dim):
                if np.random.rand() < self.step_size:
                    neighbor[i] = 1 - neighbor[i]
            if self.validate(neighbor):
                return neighbor

    def run(self):
        # Khởi tạo solution hợp lệ
        current_solution = np.random.randint(0, 2, size=self.dim)
        while not self.validate(current_solution):
            current_solution = np.random.randint(0, 2, size=self.dim)

        current_fitness = self.fitness(current_solution)
        best_solution = current_solution.copy()
        best_fitness = current_fitness

        for iteration in range(1, self.max_iter + 1):
            temp = self.initial_temp / iteration
            neighbor = self.generate_neighbor(current_solution)
            neighbor_fitness = self.fitness(neighbor)

            # Acceptance criterion
            if neighbor_fitness > current_fitness or np.random.rand() < np.exp((neighbor_fitness - current_fitness) / temp):
                current_solution = neighbor
                current_fitness = neighbor_fitness
                if current_fitness > best_fitness:
                    best_solution = current_solution
                    best_fitness = current_fitness

            if self.verbose and (iteration % 10 == 0 or iteration == self.max_iter):
                print(f"Iteration {iteration}: best fitness = {best_fitness:.6f}")

        return best_solution