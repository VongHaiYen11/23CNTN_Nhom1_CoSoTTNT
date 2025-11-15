import numpy as np


class ParticleSwarmOptimization:
    def __init__(
        self,
        fitness_func,
        lower_bound,
        upper_bound,
        dim,
        population_size=30,
        max_iter=1000,
        w=0.7,
        c1=1.5,
        c2=1.5,
        seed=None,
        verbose=False
    ):
        if seed is not None:
            np.random.seed(seed)

        self.fitness_func = fitness_func
        self.lower_bound = np.array(lower_bound)
        self.upper_bound = np.array(upper_bound)
        self.dim = dim
        self.population_size = population_size
        self.max_iter = max_iter
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.seed = seed
        self.verbose = verbose

        self.best_solution = None
        self.best_fitness = None
        self.history = []

    def simple_bounds(self, x):
        return np.clip(x, self.lower_bound, self.upper_bound)

    def update_personal_best(self, positions, pbest, pbest_values):
        for i in range(self.population_size):
            fitness = self.fitness_func(positions[i])
            if fitness < pbest_values[i]:
                pbest[i] = positions[i]
                pbest_values[i] = fitness
        return pbest, pbest_values

    def update_global_best(self, pbest, pbest_values, gbest, gbest_value):
        best_idx = np.argmin(pbest_values)
        if pbest_values[best_idx] < gbest_value:
            gbest = pbest[best_idx].copy()
            gbest_value = pbest_values[best_idx]
        return gbest, gbest_value

    def update_velocity(self, velocities, positions, pbest, gbest):
        r1 = np.random.rand(self.population_size, self.dim)
        r2 = np.random.rand(self.population_size, self.dim)
        new_velocities = (
            self.w * velocities
            + self.c1 * r1 * (pbest - positions)
            + self.c2 * r2 * (gbest - positions)
        )
        return new_velocities

    def update_position(self, positions, velocities):
        new_positions = positions + velocities
        return self.simple_bounds(new_positions)

    def run(self):
        if self.verbose:
            print("\n===== Start PSO =====")
        rng = np.random.default_rng(self.seed)

        positions = rng.uniform(
            self.lower_bound,
            self.upper_bound,
            (self.population_size, self.dim)
        )
        velocities = np.zeros((self.population_size, self.dim))
        pbest = positions.copy()
        pbest_values = np.array([self.fitness_func(x) for x in positions])
        self.best_solution = positions[np.argmin(pbest_values)].copy()
        self.best_fitness = np.min(pbest_values)
        self.history = [self.best_fitness]

        for t in range(self.max_iter):
            pbest, pbest_values = self.update_personal_best(positions, pbest, pbest_values)
            gbest, gbest_value = self.update_global_best(pbest, pbest_values, self.best_solution, self.best_fitness)
            self.best_solution = gbest.copy()
            self.best_fitness = gbest_value
            velocities = self.update_velocity(velocities, positions, pbest, gbest)
            positions = self.update_position(positions, velocities)
            self.history.append(self.best_fitness)

            if self.verbose and (t % 10 == 0 or t == self.max_iter - 1):
                print(f"Iteration {t}/{self.max_iter}: best fitness = {self.best_fitness:.6f}")

        if self.verbose:
            print("--- Optimization Results (PSO) ---")
            print(f"Best Fitness: {self.best_fitness:.6f}")
            print(f"Best Solution: {self.best_solution}")

        return self.best_solution, self.best_fitness, self.history


class ParitcleSwarmKnapsack:
    def __init__(
        self,
        weights,
        values,
        capacity,
        dim=None,
        population_size=30,
        max_iter=1000,
        w=0.7,
        c1=1.5,
        c2=1.5,
        seed=None,
        verbose=True
    ):
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

        self.best_solution = None
        self.best_fitness = 0
        self.history = []

    def is_valid(self, solution):
        return np.sum(solution * self.weights) <= self.capacity

    def repair(self, sol):
        while not self.is_valid(sol):
            ones = np.where(sol == 1)[0]
            if len(ones) == 0:
                break
            sol[np.random.choice(ones)] = 0
        return sol

    def fitness(self, solution):
        return np.sum(solution * self.values)

    def sigmoid_solution(self, x):
        s = 1 / (1 + np.exp(-x))
        sol = (s > 0.5).astype(int)
        return self.repair(sol)

    def initialize_population(self):
        positions = np.random.uniform(-4, 4, (self.population_size, self.dim))
        velocities = np.zeros((self.population_size, self.dim))
        return positions, velocities

    def run(self):
        if self.verbose:
            print("\n===== Start PSO Knapsack =====")
        positions, velocities = self.initialize_population()

        binary_positions = np.array([self.sigmoid_solution(x) for x in positions])
        fitness = np.array([self.fitness(x) for x in binary_positions])

        pbest = positions.copy()
        pbest_fitness = fitness.copy()
        gbest_idx = np.argmax(fitness)
        gbest = positions[gbest_idx].copy()
        gbest_fitness = fitness[gbest_idx]
        self.best_solution = self.sigmoid_solution(gbest)
        self.best_fitness = self.fitness(self.best_solution)
        self.history = [self.best_fitness]

        for t in range(1, self.max_iter + 1):
            r1 = np.random.rand(self.population_size, self.dim)
            r2 = np.random.rand(self.population_size, self.dim)
            velocities = (
                self.w * velocities
                + self.c1 * r1 * (pbest - positions)
                + self.c2 * r2 * (gbest - positions)
            )
            positions = positions + velocities

            binary_positions = np.array([self.sigmoid_solution(x) for x in positions])
            fitness = np.array([self.fitness(x) for x in binary_positions])

            improved = fitness > pbest_fitness
            pbest[improved] = positions[improved]
            pbest_fitness[improved] = fitness[improved]

            best_idx = np.argmax(fitness)
            if fitness[best_idx] > gbest_fitness:
                gbest = positions[best_idx].copy()
                gbest_fitness = fitness[best_idx]
                self.best_solution = self.sigmoid_solution(gbest)
                self.best_fitness = self.fitness(self.best_solution)

            self.history.append(self.best_fitness)

            if self.verbose and (t % 10 == 0 or t == self.max_iter):
                print(f"Iteration {t}/{self.max_iter}: best fitness = {self.best_fitness:.2f}")

        if self.verbose:
            print("--- Optimization Results (PSO Knapsack) ---")
            print(f"Best Fitness: {self.best_fitness:.2f}")
            print(f"Best Solution: {self.best_solution}")
            print(f"Total Weight: {np.sum(self.best_solution * self.weights):.2f}")

        return self.best_solution, self.best_fitness, self.history
