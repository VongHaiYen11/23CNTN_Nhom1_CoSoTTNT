import numpy as np


class HillClimbing:
    """Hill Climbing algorithm for continuous optimization."""
    
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
        """Initialize Hill Climbing algorithm.

        Parameters:
        fitness_func (callable): Objective function to minimize
        lower_bound (float or array): Lower bounds for each dimension
        upper_bound (float or array): Upper bounds for each dimension
        dim (int): Problem dimension
        max_iter (int): Maximum number of iterations
        n_neighbors (int): Number of neighbors to generate per iteration
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
        self.max_iter = max_iter
        self.n_neighbors = n_neighbors
        self.verbose = verbose

        self.best_solution = None
        self.best_fitness = None
        self.history = []

    def generate_neighbors(self, solution):
        """Generate neighboring solutions around current solution.

        Parameters:
        solution (np.ndarray): Current solution vector

        Returns:
        np.ndarray: Array of neighbor solutions
        """
        step = (self.upper_bound - self.lower_bound) * 0.1
        perturbations = np.random.uniform(
            -step,
            step,
            size=(self.n_neighbors, self.dim)
        )
        neighbors = np.clip(
            solution + perturbations,
            self.lower_bound,
            self.upper_bound
        )
        return neighbors

    def run(self):
        """Execute Hill Climbing optimization.

        Parameters:
        None

        Returns:
        tuple: (best_solution, best_fitness, history) where history is list of best fitness per iteration
        """
        if self.verbose:
            print("\n===== Start HC =====")
        current_solution = np.random.uniform(
            self.lower_bound,
            self.upper_bound,
            size=self.dim
        )
        current_fitness = self.fitness_func(current_solution)
        self.best_solution = current_solution.copy()
        self.best_fitness = current_fitness
        self.history = [current_fitness]

        for iteration in range(self.max_iter):
            neighbors = self.generate_neighbors(current_solution)
            fitness_values = np.apply_along_axis(self.fitness_func, 1, neighbors)
            best_idx = np.argmin(fitness_values)
            best_neighbor = neighbors[best_idx]
            best_neighbor_fitness = fitness_values[best_idx]

            if best_neighbor_fitness < current_fitness:
                current_solution = best_neighbor
                current_fitness = best_neighbor_fitness
                self.best_solution = current_solution.copy()
                self.best_fitness = current_fitness

            self.history.append(self.best_fitness)
            if self.verbose and (iteration % 10 == 0 or iteration == self.max_iter - 1):
                print(f"Iteration {iteration}/{self.max_iter}: best fitness = {self.best_fitness:.6f}")

        if self.verbose:
            print("--- Optimization Results (HC) ---")
            print(f"Best Fitness: {self.best_fitness:.6f}")
            print(f"Best Solution: {self.best_solution}")

        return self.best_solution, self.best_fitness, self.history


class HillClimbingKnapsack:
    """Hill Climbing algorithm for knapsack problem."""
    
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
        """Initialize Hill Climbing for knapsack problem.

        Parameters:
        weights (np.ndarray): Item weights
        values (np.ndarray): Item values
        capacity (float): Maximum weight capacity
        dim (int, optional): Number of items, defaults to len(weights)
        max_iter (int): Maximum number of iterations
        n_neighbors (int): Number of neighbors to generate per iteration
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
        self.max_iter = max_iter
        self.n_neighbors = n_neighbors
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

    def generate_neighbors(self, solution):
        """Generate neighboring solutions by flipping bits (knapsack version).

        Parameters:
        solution (np.ndarray): Current binary solution vector

        Returns:
        list: List of valid neighbor solutions
        """
        neighbors = []
        for _ in range(self.n_neighbors):
            neighbor = solution.copy()
            i = np.random.randint(0, self.dim)
            neighbor[i] = 1 - neighbor[i]
            if not self.is_valid(neighbor):
                neighbor = self.repair(neighbor)
            neighbors.append(neighbor)
        return neighbors

    def run(self):
        """Execute Hill Climbing for knapsack problem.

        Parameters:
        None

        Returns:
        tuple: (best_solution, best_fitness, history) where history is list of best fitness per iteration
        """
        if self.verbose:
            print("\n===== Start HC Knapsack =====")
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

            self.history.append(self.best_fitness)
            if self.verbose and (iteration % 10 == 0 or iteration == self.max_iter - 1):
                print(f"Iteration {iteration}/{self.max_iter}: best fitness = {self.best_fitness:.2f}")

        if self.verbose:
            print("--- Optimization Results (HC Knapsack) ---")
            print(f"Best Fitness: {self.best_fitness:.2f}")
            print(f"Best Solution: {self.best_solution}")

        return self.best_solution, self.best_fitness, self.history
