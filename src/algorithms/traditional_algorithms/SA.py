import numpy as np


class SimulatedAnnealing:
    def __init__(
        self,
        fitness_func,
        lower_bound,
        upper_bound,
        dim=1,
        max_iter=100,
        step_size=0.1,
        initial_temp=100,
        seed=None,
        verbose=False
    ):
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

        self.best_solution = None
        self.best_fitness = None
        self.history = []

    def run(self):
        current_solution = np.random.uniform(
            self.lower_bound,
            self.upper_bound,
            size=self.dim
        )
        current_fitness = self.fitness_func(current_solution)
        self.best_solution = current_solution.copy()
        self.best_fitness = current_fitness
        self.history = [self.best_fitness]

        for i in range(1, self.max_iter + 1):
            temp = self.initial_temp * (1 - i / self.max_iter)
            temp = max(temp, 1e-8)

            perturbation = np.random.uniform(
                -self.step_size,
                self.step_size,
                size=self.dim
            )
            candidate = np.clip(
                current_solution + perturbation,
                self.lower_bound,
                self.upper_bound
            )
            candidate_fitness = self.fitness_func(candidate)

            accept = (
                candidate_fitness < current_fitness
                or np.random.rand() < np.exp(-(candidate_fitness - current_fitness) / temp)
            )

            if accept:
                current_solution = candidate
                current_fitness = candidate_fitness
                if current_fitness < self.best_fitness:
                    self.best_solution = current_solution.copy()
                    self.best_fitness = current_fitness

            self.history.append(self.best_fitness)

            if self.verbose and (i % 50 == 0 or i == self.max_iter):
                print(f"Iteration {i}: Temp={temp:.4f}, Best fitness={self.best_fitness:.6f}")

        if self.verbose:
            print("\n--- Optimization Results (SA) ---")
            print(f"Best Fitness: {self.best_fitness:.6f}")
            print(f"Best Solution: {self.best_solution}")

        return self.best_solution, self.best_fitness, self.history


class SimulatedAnnealingKnapsack:
    def __init__(
        self,
        weights,
        values,
        capacity,
        dim=None,
        max_iter=100,
        step_size=0.1,
        initial_temp=100,
        seed=None,
        verbose=False
    ):
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

        self.best_solution = None
        self.best_fitness = 0
        self.history = []

    def fitness(self, solution):
        return np.sum(solution * self.values)

    def is_valid(self, solution):
        return np.sum(solution * self.weights) <= self.capacity

    def repair(self, sol):
        while not self.is_valid(sol):
            ones = np.where(sol == 1)[0]
            if len(ones) == 0:
                break
            sol[np.random.choice(ones)] = 0
        return sol

    def generate_neighbor(self, solution):
        while True:
            neighbor = solution.copy()
            for i in range(self.dim):
                if np.random.rand() < self.step_size:
                    neighbor[i] = 1 - neighbor[i]
            if self.is_valid(neighbor):
                return neighbor

    def run(self):
        current_solution = np.zeros(self.dim, dtype=int)
        current_fitness = self.fitness(current_solution)
        self.best_solution = current_solution.copy()
        self.best_fitness = current_fitness
        self.history = [self.best_fitness]

        for iteration in range(1, self.max_iter + 1):
            temp = self.initial_temp / iteration

            neighbor = self.generate_neighbor(current_solution)
            neighbor_fitness = self.fitness(neighbor)

            if neighbor_fitness > current_fitness or np.random.rand() < np.exp(
                (neighbor_fitness - current_fitness) / temp
            ):
                current_solution = neighbor
                current_fitness = neighbor_fitness
                if current_fitness > self.best_fitness:
                    self.best_solution = current_solution.copy()
                    self.best_fitness = current_fitness

            self.history.append(self.best_fitness)

            if self.verbose and (iteration % 50 == 0 or iteration == self.max_iter):
                print(f"Iteration {iteration}: Temp={temp:.4f}, Best fitness={self.best_fitness:.2f}")

        if self.verbose:
            print("\n--- Optimization Results (SA Knapsack) ---")
            print(f"Best Fitness: {self.best_fitness:.2f}")
            print(f"Best Solution: {self.best_solution}")

        return self.best_solution, self.best_fitness, self.history
