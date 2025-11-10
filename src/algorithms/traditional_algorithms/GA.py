import numpy as np

class GeneticAlgorithmContinuos:
  def __init__(self, fitness_func, lower_bound, upper_bound, dim=1,
               population_size=25, alpha=0.01, max_iter=1000,
               tolerance=1e-4, elitism=2, sigma=0.1, seed=None):
    if seed is not None:
      np.random.seed(seed)

    self.fitness_func = fitness_func
    self.lower_bound = lower_bound
    self.upper_bound = upper_bound
    self.dim = dim
    self.population_size = population_size
    self.alpha = alpha
    self.max_iter = max_iter
    self.tolerance = tolerance
    self.elitism = elitism
    self.sigma = sigma
    self.population = np.random.uniform(lower_bound, upper_bound,
                                        (population_size, dim))

  def evaluate_population(self):
    return np.array([self.fitness_func(ind) for ind in self.population])

  def run(self):
    for i in range(self.max_iter):
      fitness_values = self.evaluate_population()
      sorted_idx = np.argsort(fitness_values)
      self.population = self.population[sorted_idx]
      best_fitness = fitness_values[sorted_idx[0]]

      print(f"Iteration {i + 1}: best fitness = {best_fitness:.6f}")
      if best_fitness < self.tolerance:
        break

      # ---- Elitism ----
      elites = self.population[:self.elitism]

      # ---- Selection (tournament 2-way, vectorized) ----
      idx1 = np.random.randint(0, self.population_size, self.population_size)
      idx2 = np.random.randint(0, self.population_size, self.population_size)
      better = fitness_values[idx1] < fitness_values[idx2]
      selected = np.where(better[:, None], self.population[idx1], self.population[idx2])

      # ---- Crossover ----
      n_crossover = int(self.population_size * 0.4)
      parents_idx = np.random.randint(0, len(selected), (n_crossover, 2))
      dads = selected[parents_idx[:, 0]]
      moms = selected[parents_idx[:, 1]]
      crossed = self.alpha * dads + (1 - self.alpha) * moms

      # ---- Mutation ----
      n_mutation = self.population_size - len(elites) - len(crossed)
      parents = selected[np.random.randint(0, len(selected), n_mutation)]
      mutations = self.sigma * np.random.randn(n_mutation, self.dim)
      mutated = np.clip(parents + mutations, self.lower_bound, self.upper_bound)

      # ---- New population ----
      self.population = np.vstack((elites, crossed, mutated))

    best_solution = self.population[0]
    return best_solution, self.fitness_func(best_solution)
