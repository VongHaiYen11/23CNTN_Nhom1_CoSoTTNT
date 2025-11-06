import numpy as np
import math

class CuckooSearch:
    def __init__(self, fitness_func, lower_bound, upper_bound, dim=1,
                 population_size=25, alpha=1, beta=1.5,
                 pa=0.25, max_iter=100, error=1e-4, seed=None):
        if seed is not None:
            np.random.seed(seed)

        self.fitness_func = fitness_func
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.dim = dim
        self.population_size = population_size
        self.alpha = alpha
        self.beta = beta
        self.pa = pa
        self.max_iter = max_iter
        self.error = error

    def levy_flight(self):
        """Generate step vector using Lévy flight distribution"""
        beta = self.beta
        if beta <= 0.3 or beta > 2:
            beta = 1.5
        sigma_u = (math.gamma(1 + beta) * math.sin(math.pi * beta / 2) /
                   (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
        u = np.random.normal(0, sigma_u, size=self.dim)
        v = np.random.normal(0, 1, size=self.dim)
        return u / (np.abs(v) ** (1 / beta))

    def get_new_solution(self, best_solution):
        """Generate new solution around the best one"""
        new_sol = best_solution + self.alpha * self.levy_flight()
        return np.clip(new_sol, self.lower_bound, self.upper_bound)

    def run(self):
        """Run the main Cuckoo Search algorithm"""
        nests = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        fitness = np.array([self.fitness_func(x) for x in nests])

        best_solution = nests[0]
        best_fitness = self.fitness_func(best_solution)

        t = 0
        while t < self.max_iter and best_fitness > self.error:
            # Sort nests once per iteration
            sorted_idx = np.argsort(fitness)
            nests = nests[sorted_idx]
            fitness = fitness[sorted_idx]

            # Update best_solution if population contains a better one
            if fitness[0] < best_fitness:
                best_solution = nests[0].copy()
                best_fitness = fitness[0]

            # Pick a random nest to replace
            i = np.random.randint(0, self.population_size)

            # Generate new candidate from current best
            new_sol = self.get_new_solution(best_solution)
            new_fitness = self.fitness_func(new_sol)

            # Greedy replacement
            if new_fitness < fitness[i]:
                nests[i] = new_sol
                fitness[i] = new_fitness

                # Update best if better
                if new_fitness < best_fitness:
                    best_solution = new_sol.copy()
                    best_fitness = new_fitness

            # Abandon fraction pa of worst nests
            n_replace = int(self.pa * self.population_size)
            for j in range(1, n_replace + 1):
                idx = -j
                nests[idx] = np.random.uniform(self.lower_bound, self.upper_bound, self.dim)
                fitness[idx] = self.fitness_func(nests[idx])

            t += 1

        # Final sort
        sorted_idx = np.argsort(fitness)
        nests = nests[sorted_idx]
        fitness = fitness[sorted_idx]

        return nests[0], float(fitness[0])

