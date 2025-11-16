import numpy as np
import math

class CuckooSearch:
    """Cuckoo Search algorithm for continuous optimization."""
    
    def __init__(
        self,
        fitness_func,
        lower_bound,
        upper_bound,
        dim=1,
        population_size=25,
        pa=0.25,
        alpha=0.01,
        beta=1.5,
        max_iter=1000,
        seed=None,
        verbose=False
    ):
        """Initialize Cuckoo Search algorithm.

        Parameters:
        fitness_func (callable): Objective function to minimize
        lower_bound (float or array): Lower bounds for each dimension
        upper_bound (float or array): Upper bounds for each dimension
        dim (int): Problem dimension
        population_size (int): Number of nests
        pa (float): Probability of abandoning worst nests
        alpha (float): Step size scaling factor
        beta (float): Levy flight parameter
        max_iter (int): Maximum number of iterations
        seed (int, optional): Random seed for reproducibility
        verbose (bool): Whether to print progress information

        Returns:
        None
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
        self.verbose = verbose

        self.best_solution = None
        self.best_fitness = None
        self.history = []

    def levy_flight(self):
        beta = self.beta
        sigma = (
            (math.gamma(1 + beta) * math.sin(math.pi * beta / 2))
            / (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))
        ) ** (1 / beta)
        u = np.random.normal(0, sigma, self.dim)
        v = np.random.normal(0, 1, self.dim)
        step = u / (np.abs(v) ** (1 / beta))
        return step

    def get_best_nest(self, nests, new_nests, fitness):
        """Update nests with better solutions and return best nest.

        Parameters:
        nests (np.ndarray): Current nest positions
        new_nests (np.ndarray): New nest positions
        fitness (np.ndarray): Current fitness values

        Returns:
        tuple: (updated_nests, updated_fitness, best_nest, best_fitness)
        """
        new_fitness = np.array([self.fitness_func(x) for x in new_nests])
        improved = new_fitness < fitness
        fitness[improved] = new_fitness[improved]
        nests[improved] = new_nests[improved]
        best_idx = np.argmin(fitness)
        return nests, fitness, nests[best_idx], fitness[best_idx]

    def empty_nests(self, nests):
        """Abandon worst nests and generate new ones.

        Parameters:
        nests (np.ndarray): Current nest positions

        Returns:
        np.ndarray: New nest positions after abandonment
        """
        n = self.population_size
        K = np.random.rand(n, self.dim) > self.pa
        perm1 = np.random.permutation(n)
        perm2 = np.random.permutation(n)
        stepsize = np.random.rand() * (nests[perm1, :] - nests[perm2, :])
        new_nests = nests + stepsize * K
        return np.clip(new_nests, self.lower_bound, self.upper_bound)

    def get_cuckoos(self, nests, best):
        n = self.population_size
        new_nests = nests.copy()
        for j in range(n):
            s = nests[j, :].copy()
            step = self.levy_flight()
            stepsize = self.alpha * step * (s - best)
            s = s + stepsize * np.random.randn(self.dim)
            new_nests[j, :] = np.clip(s, self.lower_bound, self.upper_bound)
        return new_nests

    def run(self):
        """Execute Cuckoo Search optimization.

        Parameters:
        None

        Returns:
        tuple: (best_solution, best_fitness, history) where history is list of best fitness per iteration
        """
        if self.verbose:
            print("\n===== Start Cuckoo =====")
        n = self.population_size
        nests = np.random.uniform(
            self.lower_bound,
            self.upper_bound,
            (n, self.dim)
        )
        fitness = np.array([self.fitness_func(x) for x in nests])
        best_idx = np.argmin(fitness)
        self.best_solution = nests[best_idx].copy()
        self.best_fitness = fitness[best_idx]
        self.history = [self.best_fitness]
        t = 0
        while t < self.max_iter:
            new_nests = self.get_cuckoos(nests, self.best_solution)
            nests, fitness, best, fmin = self.get_best_nest(nests, new_nests, fitness)
            self.best_solution = best.copy()
            self.best_fitness = fmin
            new_nests = self.empty_nests(nests)
            nests, fitness, best, fmin = self.get_best_nest(nests, new_nests, fitness)
            self.best_solution = best.copy()
            self.best_fitness = fmin
            self.history.append(self.best_fitness)

            if self.verbose and (t % 10 == 0 or t == self.max_iter - 1):
                print(f"Iteration {t}/{self.max_iter}: best fitness = {self.best_fitness:.6f}")

            t += 1

        if self.verbose:
            print("--- Optimization Results (Cuckoo) ---")
            print(f"Best Fitness: {self.best_fitness:.6f}")
            print(f"Best Solution: {self.best_solution}")

        return self.best_solution, self.best_fitness, self.history


class CuckooSearchKnapsack:
    """Cuckoo Search algorithm for knapsack problem."""
    
    def __init__(
        self,
        weights,
        values,
        capacity,
        dim=None,
        population_size=25,
        pa=0.25,
        alpha=0.01,
        beta=1.5,
        max_iter=1000,
        seed=None,
        verbose=True
    ):
        """Initialize Cuckoo Search for knapsack problem.

        Parameters:
        weights (np.ndarray): Item weights
        values (np.ndarray): Item values
        capacity (float): Maximum weight capacity
        dim (int, optional): Number of items, defaults to len(weights)
        population_size (int): Number of nests
        pa (float): Probability of abandoning worst nests
        alpha (float): Step size scaling factor
        beta (float): Levy flight parameter
        max_iter (int): Maximum number of iterations
        seed (int, optional): Random seed for reproducibility
        verbose (bool): Whether to print progress information

        Returns:
        None
        """
        if seed is not None:
            np.random.seed(seed)

        self.weights = np.array(weights)
        self.values = np.array(values)
        self.capacity = capacity
        self.dim = dim if dim is not None else len(weights)
        self.population_size = population_size
        self.pa = pa
        self.alpha = alpha
        self.beta = beta
        self.max_iter = max_iter
        self.verbose = verbose

        self.best_solution = None
        self.best_fitness = 0
        self.history = []

    def fitness(self, solution):
        """Calculate fitness (total value) of solution.

        Parameters:
        solution (np.ndarray): Binary solution vector

        Returns:
        float: Total value of selected items
        """
        return np.sum(solution * self.values)

    def is_valid(self, solution):
        """Check if solution satisfies weight constraint.

        Parameters:
        solution (np.ndarray): Binary solution vector

        Returns:
        bool: True if solution is valid, False otherwise
        """
        return np.sum(solution * self.weights) <= self.capacity

    def get_random_nest(self):
        """Generate random valid nest (solution).

        Parameters:
        None

        Returns:
        np.ndarray: Random valid binary solution
        """
        while True:
            nest = np.random.randint(0, 2, size=self.dim)
            if self.is_valid(nest):
                return nest

    def initialize_population(self):
        """Initialize population with random valid nests.

        Parameters:
        None

        Returns:
        np.ndarray: Array of valid binary solutions
        """
        return np.array([self.get_random_nest() for _ in range(self.population_size)])

    def levy_flight(self):
        """Generate Levy flight step for knapsack.

        Parameters:
        None

        Returns:
        np.ndarray: Levy flight step vector
        """
        beta = self.beta
        sigma = (
            (math.gamma(1 + beta) * math.sin(math.pi * beta / 2))
            / (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))
        ) ** (1 / beta)
        u = np.random.normal(0, sigma, self.dim)
        v = np.random.normal(0, 1, self.dim)
        step = u / (np.abs(v) ** (1 / beta))
        return step

    def repair(self, sol):
        """Repair invalid solution by removing items until constraint is satisfied.

        Parameters:
        sol (np.ndarray): Binary solution vector

        Returns:
        np.ndarray: Repaired solution vector
        """
        while not self.is_valid(sol):
            ones = np.where(sol == 1)[0]
            if len(ones) == 0:
                break
            sol[np.random.choice(ones)] = 0
        return sol

    def sigmoid_solution(self, x):
        """Convert continuous position to binary solution using sigmoid function.

        Parameters:
        x (np.ndarray): Continuous position vector

        Returns:
        np.ndarray: Binary solution vector
        """
        s = 1 / (1 + np.exp(-x))
        sol = (s > 0.5).astype(int)
        return sol

    def get_best_nest(self, nests, new_nests, fitness):
        """Update nests with better solutions and return best nest (knapsack version).

        Parameters:
        nests (np.ndarray): Current nest positions
        new_nests (np.ndarray): New nest positions
        fitness (np.ndarray): Current fitness values

        Returns:
        tuple: (updated_nests, updated_fitness, best_nest, best_fitness)
        """
        new_fitness = np.array([self.fitness(x) for x in new_nests])
        improved = new_fitness > fitness
        for i in range(self.population_size):
            if improved[i]:
                nests[i] = new_nests[i].copy()
                fitness[i] = new_fitness[i]
        best_idx = np.argmax(fitness)
        return nests, fitness, nests[best_idx].copy(), fitness[best_idx]

    def empty_nests(self, nests):
        """Abandon worst nests and generate new valid ones (knapsack version).

        Parameters:
        nests (np.ndarray): Current nest positions

        Returns:
        np.ndarray: New nest positions after abandonment
        """
        new_nests = nests.copy()
        for i in range(self.population_size):
            if np.random.rand() < self.pa:
                new_nests[i] = self.get_random_nest()
        return new_nests

    def get_cuckoos(self, nests, best):
        """Generate new solutions using Levy flight for knapsack.

        Parameters:
        nests (np.ndarray): Current nest positions
        best (np.ndarray): Best nest position

        Returns:
        np.ndarray: New nest positions after Levy flight and repair
        """
        n = self.population_size
        new_nests = nests.copy()
        for j in range(n):
            s = nests[j, :].copy()
            step = self.levy_flight()
            stepsize = self.alpha * step * (s - best)
            s = s + stepsize * np.random.randn(self.dim)
            s = self.repair(self.sigmoid_solution(s))
            new_nests[j, :] = s
        return new_nests

    def run(self):
        """Execute Cuckoo Search for knapsack problem.

        Parameters:
        None

        Returns:
        tuple: (best_solution, best_fitness, history) where history is list of best fitness per iteration
        """
        if self.verbose:
            print("\n===== Start Cuckoo Knapsack =====")
        nests = self.initialize_population()
        fitness = np.array([self.fitness(x) for x in nests])
        best_idx = np.argmax(fitness)
        self.best_solution = nests[best_idx].copy()
        self.best_fitness = fitness[best_idx]
        self.history = [self.best_fitness]
        for t in range(1, self.max_iter + 1):
            new_nests = self.get_cuckoos(nests, self.best_solution)
            nests, fitness, best, fmax = self.get_best_nest(nests, new_nests, fitness)
            self.best_solution = best.copy()
            self.best_fitness = fmax
            new_nests = self.empty_nests(nests)
            nests, fitness, best, fmax = self.get_best_nest(nests, new_nests, fitness)
            self.best_solution = best.copy()
            self.best_fitness = fmax
            self.history.append(self.best_fitness)

            if self.verbose and (t % 10 == 0 or t == self.max_iter):
                print(f"Iteration {t}/{self.max_iter}: best fitness = {self.best_fitness:.2f}")

        if self.verbose:
            print("--- Optimization Results (Cuckoo Knapsack) ---")
            print(f"Best Fitness: {self.best_fitness:.2f}")
            print(f"Best Solution: {self.best_solution}")

        return self.best_solution, self.best_fitness, self.history
