import numpy as np

class SimulatedAnnealing:
  def __init__(self, fitness_func, lower_bound, upper_bound, dim=1,
               max_iter=100, step_size=0.1, initial_temp=100, seed=None, verbose=False):
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

    # Khởi tạo nghiệm ban đầu
    self.current_solution = np.zeros(self.dim)
    self.current_fitness = self.fitness_func(self.current_solution)
    self.best_solution = self.current_solution.copy()
    self.best_fitness = self.current_fitness

  def run(self):
    for i in range(1, self.max_iter + 1):
      temp = self.initial_temp * (1 - i / self.max_iter)
      if temp < 1e-8:
        temp = 1e-8

      # Sinh neighbor
      perturbation = np.random.uniform(-self.step_size, self.step_size, size=self.dim)
      candidate = np.clip(self.current_solution + perturbation, self.lower_bound, self.upper_bound)
      candidate_fitness = self.fitness_func(candidate)

      # Quyết định chấp nhận
      accept = (
        candidate_fitness < self.current_fitness
        or np.random.rand() < np.exp(-(candidate_fitness - self.current_fitness) / temp)
      )

      if accept:
        self.current_solution = candidate
        self.current_fitness = candidate_fitness
        if self.current_fitness < self.best_fitness:
          self.best_solution = self.current_solution.copy()
          self.best_fitness = self.current_fitness

      if self.verbose:
        print(f"Iteration {i}: Temp={temp:.4f}, Best fitness={self.best_fitness:.6f}")

    print(self.best_solution, self.best_fitness)
    return self.best_solution, self.best_fitness
