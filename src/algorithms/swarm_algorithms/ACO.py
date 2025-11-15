import numpy as np


class AntColonyOptimizationContinuous:
    def __init__(
        self,
        fitness_func,
        dim,
        lower_bound,
        upper_bound,
        population_size=30,
        max_iter=100,
        archive_size=None,
        rho=0.85,
        seed=None,
        verbose=False
    ):
        self.fitness_func = fitness_func
        self.dim = dim
        self.population_size = population_size
        self.max_iter = max_iter
        self.rng = np.random.default_rng(seed)

        self.lower_bound = np.array(lower_bound)
        self.upper_bound = np.array(upper_bound)
        self.rho = rho

        if archive_size is None:
            self.archive_size = self.population_size
        else:
            self.archive_size = archive_size

        self.archive = self.rng.uniform(
            self.lower_bound,
            self.upper_bound,
            (self.archive_size, self.dim)
        )
        self.archive_fitness = np.array([self.fitness_func(s) for s in self.archive])
        self.sort_archive()

        self.archive_weights = self.calculate_acor_weights()

        self.best_solution = self.archive[0].copy()
        self.best_fitness = self.archive_fitness[0]
        self.history = []
        self.verbose = verbose

    def sort_archive(self):
        sort_indices = np.argsort(self.archive_fitness)
        self.archive = self.archive[sort_indices]
        self.archive_fitness = self.archive_fitness[sort_indices]

    def calculate_acor_weights(self):
        k = self.archive_size
        ranks = np.arange(k)
        q = 0.05
        weights = (
            (1 / (q * k * np.sqrt(2 * np.pi)))
            * np.exp(-ranks**2 / (2 * q**2 * k**2))
        )
        return weights / np.sum(weights)

    def construct_solutions(self):
        all_solutions = np.zeros((self.population_size, self.dim))
        for k in range(self.population_size):
            guide_idx = self.rng.choice(self.archive_size, p=self.archive_weights)
            guide_sol = self.archive[guide_idx]

            new_sol = np.zeros(self.dim)
            for d in range(self.dim):
                sigma_d = self.rho * np.mean(np.abs(self.archive[:, d] - guide_sol[d]))
                new_sol[d] = self.rng.normal(loc=guide_sol[d], scale=sigma_d)

            all_solutions[k] = np.clip(new_sol, self.lower_bound, self.upper_bound)
        return all_solutions

    def update_archive(self, new_solutions, new_fitness):
        combined_solutions = np.vstack((self.archive, new_solutions))
        combined_fitness = np.concatenate((self.archive_fitness, new_fitness))

        sort_indices = np.argsort(combined_fitness)
        self.archive = combined_solutions[sort_indices][:self.archive_size]
        self.archive_fitness = combined_fitness[sort_indices][:self.archive_size]

    def run(self):
        self.best_solution = self.archive[0].copy()
        self.best_fitness = self.archive_fitness[0]
        self.history = []

        for it in range(self.max_iter):
            all_solutions = self.construct_solutions()
            all_fitness = np.array([self.fitness_func(s) for s in all_solutions])

            self.update_archive(all_solutions, all_fitness)

            if self.archive_fitness[0] < self.best_fitness:
                self.best_fitness = self.archive_fitness[0]
                self.best_solution = self.archive[0].copy()

            self.history.append(self.best_fitness)
            if self.verbose and (it % 10 == 0 or it == self.max_iter - 1):
                print(f"Iter {it+1}/{self.max_iter}: best fitness = {self.best_fitness:.6f}")

        if self.verbose:
            print("\n--- Optimization Results (ACO Continuous) ---")
            print(f"Best Fitness: {self.best_fitness:.6f}")
            print(f"Best Solution: {self.best_solution}")

        return self.best_solution, self.best_fitness, self.history


class AntColonyOptimizationKnapsack:
    def __init__(
        self,
        weights,
        values,
        capacity,
        n_ants=30,
        max_iter=100,
        alpha=1.0,
        beta=2.0,
        rho=0.3,
        Q=1.0,
        seed=None,
        verbose=True
    ):
        self.weights = np.array(weights)
        self.values = np.array(values)
        self.capacity = capacity
        self.n_items = len(weights)

        self.n_ants = n_ants
        self.max_iter = max_iter
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.Q = Q
        self.verbose = verbose

        self.rng = np.random.default_rng(seed)

        self.tau = np.ones(self.n_items)
        self.mu = self.values / (self.weights + 1e-9)

        self.best_solution = None
        self.best_fitness = 0
        self.history = []

    def fitness(self, solution):
        total_weight = np.sum(solution * self.weights)
        total_value = np.sum(solution * self.values)
        if total_weight > self.capacity:
            return 0
        return total_value

    def probabilities(self, allowed):
        tau_allowed = self.tau[allowed] ** self.alpha
        mu_allowed = self.mu[allowed] ** self.beta
        numerator = tau_allowed * mu_allowed
        denom = np.sum(numerator)
        if denom == 0:
            return np.ones_like(numerator) / len(numerator)
        return numerator / denom

    def construct_solution(self):
        solution = np.zeros(self.n_items, dtype=int)
        current_weight = 0
        available = np.arange(self.n_items)

        while len(available) > 0:
            allowed = available[
                self.weights[available] + current_weight <= self.capacity
            ]
            if len(allowed) == 0:
                break

            probs = self.probabilities(allowed)
            chosen = self.rng.choice(allowed, p=probs)

            solution[chosen] = 1
            current_weight += self.weights[chosen]
            available = available[available != chosen]

        fitness = self.fitness(solution)
        return solution, fitness

    def update_pheromone(self, all_solutions, all_fitness):
        self.tau *= (1 - self.rho)

        for sol, fit in zip(all_solutions, all_fitness):
            if fit > 0:
                delta_tau = self.Q * fit
                self.tau += delta_tau * sol

    def run(self):
        for it in range(1, self.max_iter + 1):
            all_solutions = []
            all_fitness = []

            for _ in range(self.n_ants):
                sol, fit = self.construct_solution()
                all_solutions.append(sol)
                all_fitness.append(fit)

            all_fitness = np.array(all_fitness)
            best_idx = np.argmax(all_fitness)

            if all_fitness[best_idx] > self.best_fitness:
                self.best_fitness = all_fitness[best_idx]
                self.best_solution = all_solutions[best_idx].copy()

            self.history.append(self.best_fitness)

            self.update_pheromone(all_solutions, all_fitness)

            if self.verbose and (it % 10 == 0 or it == self.max_iter):
                print(f"Iter {it}/{self.max_iter}: best fitness = {self.best_fitness:.2f}")

        if self.verbose:
            print("\n=== Final Result (ACO-Knapsack) ===")
            print(f"Best solution: {self.best_solution}")
            print(f"Total value: {self.best_fitness:.2f}")
            print(f"Total weight: {np.sum(self.best_solution * self.weights):.2f}")

        return self.best_solution, self.best_fitness, self.history
