import numpy as np


class HillClimbing:
    def __init__(
        self,
        fitness_func,
        lower_bound,
        upper_bound,
        dim=1,
        max_iter=100,
        n_neighbors=1000,
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
        self.n_neighbors = n_neighbors
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
        self.history = [current_fitness]
        step = (self.upper_bound - self.lower_bound) * 0.1

        for iteration in range(self.max_iter):
            perturbations = np.random.uniform(
                -step,
                step,
                size=(self.n_neighbors, self.dim)
            )
            neighbors = np.clip(
                current_solution + perturbations,
                self.lower_bound,
                self.upper_bound
            )
            fitness_values = np.apply_along_axis(self.fitness_func, 1, neighbors)
            best_idx = np.argmin(fitness_values)
            best_neighbor = neighbors[best_idx]
            best_neighbor_fitness = fitness_values[best_idx]

            if best_neighbor_fitness < current_fitness:
                current_solution = best_neighbor
                current_fitness = best_neighbor_fitness
                self.best_solution = current_solution.copy()
                self.best_fitness = current_fitness
            else:
                break

            self.history.append(self.best_fitness)
            if self.verbose:
                print(f"Iteration {iteration+1}: best fitness = {self.best_fitness:.6f}")

        if self.verbose:
            print("\n--- Optimization Results (Hill Climbing) ---")
            print(f"Best Fitness: {self.best_fitness:.6f}")
            print(f"Best Solution: {self.best_solution}")

        return self.best_solution, self.best_fitness, self.history


class HillClimbingKnapsack:
    def __init__(
        self,
        weights,
        values,
        capacity,
        dim=None,
        max_iter=100,
        n_neighbors=10,
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
        self.n_neighbors = n_neighbors
        self.verbose = verbose

        self.best_solution = None
        self.best_fitness = 0
        self.history = []

    def fitness(self, solution):
        return np.sum(solution * self.values)

    def is_valid(self, solution):
        return np.sum(solution * self.weights) <= self.capacity

    def generate_neighbors(self, solution):
        neighbors = []
        for _ in range(self.n_neighbors):
            while True:
                neighbor = solution.copy()
                i = np.random.randint(0, self.dim)
                neighbor[i] = 1 - neighbor[i]
                if self.is_valid(neighbor):
                    neighbors.append(neighbor)
                    break
        return neighbors

    def run(self):
        self.best_solution = np.zeros(self.dim, dtype=int)
        while not self.is_valid(self.best_solution):
            self.best_solution = np.random.randint(0, 2, self.dim)
        self.best_fitness = self.fitness(self.best_solution)
        self.history = [self.best_fitness]

        for iteration in range(self.max_iter):
            neighbors = self.generate_neighbors(self.best_solution)
            if not neighbors:
                break

            fitness_values = np.array([self.fitness(n) for n in neighbors])
            idx_best = np.argmax(fitness_values)
            best_neighbor = neighbors[idx_best]
            best_neighbor_fitness = fitness_values[idx_best]

            if best_neighbor_fitness > self.best_fitness:
                self.best_solution = best_neighbor.copy()
                self.best_fitness = best_neighbor_fitness
            else:
                break

            self.history.append(self.best_fitness)
            if self.verbose:
                print(f"Iteration {iteration+1}: best fitness = {self.best_fitness:.2f}")

        if self.verbose:
            print("\n--- Optimization Results (Hill Climbing Knapsack) ---")
            print(f"Best Fitness: {self.best_fitness:.2f}")
            print(f"Best Solution: {self.best_solution}")

        return self.best_solution, self.best_fitness, self.history
