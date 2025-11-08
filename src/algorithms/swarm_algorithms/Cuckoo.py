import numpy as np
import math

class CuckooSearch:
    def __init__(self, fitness_func, lower_bound, upper_bound, dim=1,
                 population_size=25, pa=0.25, alpha=0.01, beta=1.5,
                 max_iter=1000, tolerance=1e-4, seed=None):
        """
        Improved Cuckoo Search (Yang & Deb, 2009) - Vectorized version
        """
        if seed is not None:
            np.random.seed(seed)

        self.fitness_func = fitness_func
        self.lower_bound = np.array(lower_bound)
        self.upper_bound = np.array(upper_bound)
        self.dim = dim
        self.population_size = population_size
        self.pa = pa
        self.alpha = alpha
        self.beta = beta
        self.max_iter = max_iter
        self.tolerance = tolerance

    def levy_flight(self, size):
        """Generate Lévy flight step using Mantegna’s algorithm."""
        beta = self.beta
        sigma = (math.gamma(1 + beta) * math.sin(math.pi * beta / 2) /
                 (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
        u = np.random.normal(0, sigma, size)
        v = np.random.normal(0, 1, size)
        return u / (np.abs(v) ** (1 / beta))

    def get_best_nest(self, nests, new_nests, fitness):
        """Evaluate and update nests if new ones are better."""
        new_fitness = np.array([self.fitness_func(x) for x in new_nests])
        improved = new_fitness < fitness
        fitness[improved] = new_fitness[improved]
        nests[improved] = new_nests[improved]
        best_idx = np.argmin(fitness)
        return nests, fitness, nests[best_idx], fitness[best_idx]

    def empty_nests(self, nests):
        """Replace a fraction pa of worst nests (discovery)."""
        n = self.population_size
        K = np.random.rand(n, self.dim) > self.pa
        perm1 = np.random.permutation(n)
        perm2 = np.random.permutation(n)
        stepsize = np.random.rand() * (nests[perm1, :] - nests[perm2, :])
        new_nests = nests + stepsize * K
        return np.clip(new_nests, self.lower_bound, self.upper_bound)

    def get_cuckoos(self, nests, best):
        """Generate new solutions via Lévy flight."""
        n = self.population_size
        steps = self.levy_flight((n, self.dim))
        stepsize = self.alpha * steps * (nests - best)
        new_nests = nests + stepsize * np.random.randn(n, self.dim)
        return np.clip(new_nests, self.lower_bound, self.upper_bound)
    
    def run(self, verbose=False):
        """Main optimization loop."""
        n = self.population_size
        nests = np.random.uniform(self.lower_bound, self.upper_bound, (n, self.dim))
        fitness = np.array([self.fitness_func(x) for x in nests])

        best_idx = np.argmin(fitness)
        best = nests[best_idx].copy()
        fmin = fitness[best_idx]
        t = 0
        while t < self.max_iter and fmin > self.tolerance:
            # Lévy flight phase
            new_nests = self.get_cuckoos(nests, best)
            nests, fitness, best, fmin = self.get_best_nest(nests, new_nests, fitness)

            # Empty nest phase
            new_nests = self.empty_nests(nests)
            nests, fitness, best, fmin = self.get_best_nest(nests, new_nests, fitness)

            if verbose and (t % 50 == 0 or t == self.max_iter - 1):
                print(f"Iteration {t+1}/{self.max_iter}: best fitness = {fmin:.6f}")

            t += 1

        return best, fmin
