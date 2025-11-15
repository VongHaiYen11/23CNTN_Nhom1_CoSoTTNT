import numpy as np


class FireflyAlgorithm:
    def __init__(
        self,
        fitness_func,
        lower_bound=-5.12,
        upper_bound=5.12,
        dim=30,
        population_size=100,
        max_iter=500,
        alpha=0.5,
        beta0=1.0,
        gamma=0.01,
        seed=None,
        verbose=False
    ):
        self.fireflies = None
        if seed is not None:
            np.random.seed(seed)

        self.fitness_func = fitness_func
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.dim = dim
        self.population_size = population_size
        self.max_iter = max_iter
        self.alpha = alpha
        self.beta0 = beta0
        self.gamma = gamma
        self.verbose = verbose

        self.best_solution = None
        self.best_fitness = None
        self.history = []

    def initialize_population(self):
        fireflies = (
            self.lower_bound
            + (self.upper_bound - self.lower_bound)
            * np.random.rand(self.population_size, self.dim)
        )
        intensity = np.apply_along_axis(self.fitness_func, 1, fireflies)

        best_idx = np.argmin(intensity)
        best_solution = np.copy(fireflies[best_idx])
        best_fitness = intensity[best_idx]
        fitness_history = [best_fitness]

        self.fireflies = fireflies
        self.best_solution = best_solution
        self.best_fitness = best_fitness
        self.history = fitness_history
        return fireflies, intensity, best_solution, best_fitness, fitness_history

    def move_fireflies(self, fireflies, intensity, alpha):
        for i in range(self.population_size):
            for j in range(self.population_size):
                if intensity[j] < intensity[i]:
                    r2 = np.sum((fireflies[i] - fireflies[j]) ** 2)

                    beta = self.beta0 * np.exp(-self.gamma * r2)

                    random_step = (
                        alpha
                        * (np.random.uniform(-0.5, 0.5, self.dim))
                        * (self.upper_bound - self.lower_bound)
                    )
                    fireflies[i] += beta * (fireflies[j] - fireflies[i]) + random_step

                    fireflies[i] = np.clip(
                        fireflies[i],
                        self.lower_bound,
                        self.upper_bound
                    )

                    intensity[i] = self.fitness_func(fireflies[i])

    def update_best_solution(
        self,
        fireflies,
        intensity,
        best_solution,
        best_fitness,
        fitness_history
    ):
        current_best_idx = np.argmin(intensity)
        current_best_fitness = intensity[current_best_idx]

        if current_best_fitness < best_fitness:
            best_fitness = current_best_fitness
            best_solution = np.copy(fireflies[current_best_idx])

        fitness_history.append(best_fitness)
        return best_solution, best_fitness, fitness_history

    def run(self):
        if self.verbose:
            print("\n===== Start FA =====")
        (
            fireflies,
            intensity,
            best_solution,
            best_fitness,
            fitness_history
        ) = self.initialize_population()
        alpha = self.alpha

        for iteration in range(self.max_iter):
            alpha *= 0.97

            self.move_fireflies(fireflies, intensity, alpha)

            self.best_solution, self.best_fitness, self.history = self.update_best_solution(
                fireflies,
                intensity,
                self.best_solution,
                self.best_fitness,
                self.history
            )

            if self.verbose and (iteration % 10 == 0 or iteration == self.max_iter - 1):
                print(f"Iteration {iteration}/{self.max_iter}: best fitness = {self.best_fitness:.6f}")

        self.fireflies = fireflies
        if self.verbose:
            print("--- Optimization Results (FA) ---")
            print(f"Best Fitness: {self.best_fitness:.6f}")
            print(f"Best Solution: {self.best_solution}")

        return self.best_solution, self.best_fitness, self.history


class FireflyKnapsack:
    def __init__(
        self,
        weights,
        values,
        max_weight,
        population_size=100,
        max_iter=200,
        alpha=0.2,
        beta0=1.0,
        gamma=0.01,
        seed=None,
        verbose=False
    ):
        self.weights = np.array(weights)
        self.values = np.array(values)
        self.max_weight = max_weight
        self.n_items = len(weights)
        self.population_size = population_size
        self.max_iter = max_iter
        self.alpha = alpha
        self.beta0 = beta0
        self.gamma = gamma
        self.verbose = verbose

        self.fireflies = None
        self.fitness = None

        self.best_solution = None
        self.best_fitness = 0
        self.history = []

        if seed is not None:
            np.random.seed(seed)

    def is_valid(self, solution):
        return np.sum(solution * self.weights) <= self.max_weight

    def initialize_population(self):
        fireflies = np.random.randint(0, 2, size=(self.population_size, self.n_items))
        fitness = np.array([self.evaluate(self.repair(sol)) for sol in fireflies])

        best_idx = np.argmax(fitness)
        best_solution = fireflies[best_idx].copy()
        best_fitness = fitness[best_idx]
        fitness_history = [best_fitness]

        self.fireflies = fireflies
        self.fitness = fitness
        self.best_solution = best_solution
        self.best_fitness = best_fitness
        self.history = fitness_history

        return fireflies, fitness, best_solution, best_fitness, fitness_history

    def evaluate(self, solution):
        return np.sum(solution * self.values)

    def hamming_distance(self, x, y):
        return np.sum(x != y)

    def crossover(self, parent1, parent2):
        if np.random.rand() < 0.8:
            point = np.random.randint(1, self.n_items)
            child = np.concatenate([parent1[:point], parent2[point:]])
        else:
            child = parent1.copy()
        return child

    def mutate(self, solution):
        for i in range(self.n_items):
            if np.random.rand() < self.alpha:
                solution[i] = 1 - solution[i]
        return solution

    def repair(self, solution):
        sol = solution.copy()
        while not self.is_valid(sol):
            ones = np.where(sol == 1)[0]
            if len(ones) == 0:
                break
            sol[np.random.choice(ones)] = 0
        return sol

    def move_firefly(self, i, j):
        if self.fitness[j] <= self.fitness[i]:
            return

        r = self.hamming_distance(self.fireflies[i], self.fireflies[j])
        beta = self.beta0 * np.exp(-self.gamma * r)

        if np.random.rand() < beta:
            child = self.crossover(self.fireflies[i], self.fireflies[j])
            child = self.mutate(child)
            child = self.repair(child)

            child_fitness = self.evaluate(child)
            if child_fitness > self.fitness[i]:
                self.fireflies[i] = child
                self.fitness[i] = child_fitness

    def run(self):
        if self.verbose:
            print("\n===== Start FA Knapsack =====")
        (
            fireflies,
            fitness,
            _,
            _,
            _
        ) = self.initialize_population()
        alpha = self.alpha

        for iteration in range(self.max_iter):
            alpha *= 0.97

            for i in range(self.population_size):
                for j in range(self.population_size):
                    if i != j:
                        self.move_firefly(i, j)

            current_best_idx = np.argmax(fitness)
            if fitness[current_best_idx] > self.best_fitness:
                self.best_fitness = fitness[current_best_idx]
                self.best_solution = fireflies[current_best_idx].copy()

            self.history.append(self.best_fitness)

            if self.verbose and (iteration % 10 == 0 or iteration == self.max_iter - 1):
                print(f"Iteration {iteration}/{self.max_iter}: best fitness = {self.best_fitness:.2f}")

        if self.verbose:
            print("--- Optimization Results (Firefly Knapsack) ---")
            total_weight = np.sum(self.best_solution * self.weights)
            print(f"Best Value: {self.best_fitness:.2f}")
            print(f"Total Weight: {total_weight:.2f} <= {self.max_weight:.2f}")
            print(f"Selected Items: {self.best_solution}")

        return self.best_solution, self.best_fitness, self.history
