import numpy as np


class GeneticAlgorithmContinuous:
    def __init__(
        self,
        fitness_func,
        lower_bound,
        upper_bound,
        dim=1,
        population_size=25,
        alpha=0.01,
        max_iter=1000,
        elitism=2,
        sigma=0.1,
        seed=None,
        verbose=False
    ):
        if seed is not None:
            np.random.seed(seed)
        self.fitness_func = fitness_func
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.dim = dim
        self.population_size = population_size
        self.alpha = alpha
        self.max_iter = max_iter
        self.elitism = elitism
        self.sigma = sigma
        self.population = np.random.uniform(
            lower_bound,
            upper_bound,
            (population_size, dim)
        )
        self.verbose = verbose

        self.best_solution = None
        self.best_fitness = None
        self.history = []

    def evaluate_population(self):
        return np.array([self.fitness_func(ind) for ind in self.population])

    def sort_population(self):
        fitness_values = self.evaluate_population()
        sorted_idx = np.argsort(fitness_values)
        self.population = self.population[sorted_idx]
        return fitness_values[sorted_idx]

    def elitism_selection(self):
        self.sort_population()
        return self.population[:self.elitism]

    def selection(self, n_select):
        fitness_values = self.evaluate_population()
        idx1 = np.random.randint(0, self.population_size, n_select)
        idx2 = np.random.randint(0, self.population_size, n_select)
        better = fitness_values[idx1] < fitness_values[idx2]
        selected = np.where(
            better[:, None],
            self.population[idx1],
            self.population[idx2]
        )
        return selected

    def crossover(self, selected, n_crossover):
        parents_idx = np.random.randint(0, len(selected), (n_crossover, 2))
        dads = selected[parents_idx[:, 0]]
        moms = selected[parents_idx[:, 1]]
        crossed = self.alpha * dads + (1 - self.alpha) * moms
        return crossed

    def mutation(self, selected, n_mutation):
        parents = selected[np.random.randint(0, len(selected), n_mutation)]
        mutations = self.sigma * np.random.randn(n_mutation, self.dim)
        mutated = np.clip(
            parents + mutations,
            self.lower_bound,
            self.upper_bound
        )
        return mutated

    def run(self):
        if self.verbose:
            print("\n===== Start GA =====")
        self.history = []
        for i in range(self.max_iter):
            fitness_values = self.sort_population()
            self.best_fitness = fitness_values[0]
            self.history.append(self.best_fitness)

            elites = self.elitism_selection()
            new_population = elites

            while len(new_population) < self.population_size:
                remaining = self.population_size - len(new_population)
                n_selection = max(int(remaining * 0.4), 2)
                n_crossover = max(int(remaining * 0.4), 1)
                selected = self.selection(n_selection)
                crossed = self.crossover(selected, n_crossover)
                n_mutation = max(remaining - len(crossed), 1)
                mutated = self.mutation(selected, n_mutation)
                combined = np.vstack((selected, crossed, mutated))
                new_population = np.vstack((new_population, combined))

            self.population = new_population[:self.population_size]

            if self.verbose and (i % 10 == 0 or i == self.max_iter - 1):
                print(f"Iteration {i}/{self.max_iter}: best fitness = {self.best_fitness:.6f}")

        self.best_solution = self.population[0]
        if self.verbose:
            print("--- Optimization Results (GA) ---")
            print(f"Best Fitness: {self.fitness_func(self.best_solution):.6f}")
            print(f"Best Solution: {self.best_solution}")
        return self.best_solution, self.fitness_func(self.best_solution), self.history


class GeneticAlgorithmKnapsack:
    def __init__(
        self,
        weights,
        values,
        capacity,
        population_size=20,
        max_iter=1000,
        elitism=2,
        mutation_rate=0.1,
        crossover_rate=0.8,
        seed=None,
        verbose=True
    ):
        if seed is not None:
            np.random.seed(seed)

        self.weights = np.array(weights)
        self.values = np.array(values)
        self.capacity = capacity
        self.dim = len(weights)
        self.population_size = population_size
        self.max_iter = max_iter
        self.elitism = elitism
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.verbose = verbose

        self.population = self.initialize_population()
        self.best_solution = np.empty((0, self.dim), dtype=int)
        self.best_fitness = 0
        self.history = []

    def initialize_population(self):
        population = []
        while len(population) < self.population_size:
            individual = np.random.randint(0, 2, self.dim)
            if self.is_valid(individual):
                population.append(individual.copy())
        return np.array(population).reshape(-1, self.dim)

    def fitness_func(self, x):
        return np.sum(x * self.values)

    def is_valid(self, x):
        return np.sum(x * self.weights) <= self.capacity

    def sort_population(self):
        fitness_values = np.array([self.fitness_func(ind) for ind in self.population])
        idx = np.argsort(-fitness_values)
        self.population = self.population[idx]
        return fitness_values[idx]

    def elitism_selection(self):
        self.sort_population()
        return self.population[:self.elitism].copy()

    def selection(self, n_select):
        selection_pool = []
        for _ in range(n_select):
            idx1, idx2 = np.random.randint(0, self.population_size, size=2)
            candidate = (
                self.population[idx1]
                if self.fitness_func(self.population[idx1]) > self.fitness_func(self.population[idx2])
                else self.population[idx2]
            )
            if self.is_valid(candidate):
                selection_pool.append(candidate.copy())
        if len(selection_pool) == 0:
            selection_pool = self.population[np.random.choice(self.population_size, size=n_select)]
        return np.array(selection_pool).reshape(-1, self.dim)

    def crossover(self, selected, n_crossover):
        crossover_pool = []
        if len(selected) < 2:
            return np.empty((0, self.dim), dtype=int)
        for _ in range(n_crossover // 2):
            idx_pair = np.random.randint(0, len(selected), size=2)
            offspring1 = selected[idx_pair[0]].copy()
            offspring2 = selected[idx_pair[1]].copy()
            for pos in range(self.dim):
                if np.random.rand() < self.crossover_rate:
                    offspring1[pos], offspring2[pos] = offspring2[pos], offspring1[pos]
            if self.is_valid(offspring1):
                crossover_pool.append(offspring1)
            if self.is_valid(offspring2):
                crossover_pool.append(offspring2)
        if len(crossover_pool) == 0:
            return np.empty((0, self.dim), dtype=int)
        return np.array(crossover_pool).reshape(-1, self.dim)

    def mutation(self, selected, n_mutation):
        mutated = []
        if len(selected) < 1:
            return np.empty((0, self.dim), dtype=int)
        for _ in range(n_mutation):
            child = selected[np.random.randint(0, len(selected))].copy()
            for pos in range(self.dim):
                if np.random.rand() < self.mutation_rate:
                    child[pos] = 1 - child[pos]
            if self.is_valid(child):
                mutated.append(child)
        if len(mutated) == 0:
            return np.empty((0, self.dim), dtype=int)
        return np.array(mutated).reshape(-1, self.dim)

    def run(self):
        if self.verbose:
            print("\n===== Start GA Knapsack =====")
        self.best_solution = np.empty((0, self.dim), dtype=int)
        self.best_fitness = 0
        self.history = []

        for t in range(self.max_iter):
            elites = self.elitism_selection()
            new_population = elites.copy()

            while len(new_population) < self.population_size:
                remaining = self.population_size - len(new_population)
                n_selection = max(int(remaining * 0.4), 2)
                n_crossover = max(int(remaining * 0.4), 1)
                selected = self.selection(n_selection)
                crossed = self.crossover(selected, n_crossover)
                n_mutation = max(remaining - len(crossed), 1)
                mutated = self.mutation(selected, n_mutation)

                combined = np.concatenate((selected, crossed, mutated), axis=0)
                combined = np.unique(combined.reshape(-1, self.dim), axis=0)
                new_population = np.concatenate((new_population, combined), axis=0)

            self.population = new_population[:self.population_size]
            self.sort_population()

            new_best_fitness = self.fitness_func(self.population[0])
            if new_best_fitness > self.best_fitness:
                self.best_fitness = new_best_fitness
                self.best_solution = self.population[0]

            self.history.append(self.best_fitness)

            if self.verbose and (t % 10 == 0 or t == self.max_iter - 1):
                print(f"Iteration {t}/{self.max_iter}: best fitness = {self.best_fitness:.2f}")

        if self.verbose:
            print("--- Optimization Results (GA Knapsack) ---")
            print(f"Best Fitness: {self.best_fitness:.2f}")
            print(f"Best Solution: {self.best_solution}")

        return self.best_solution, self.best_fitness, self.history
