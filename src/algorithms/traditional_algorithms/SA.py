import numpy as np


class SimulatedAnnealing:
    """Simulated Annealing algorithm for continuous optimization."""
    
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
        """Initialize Simulated Annealing algorithm.

        Parameters:
        fitness_func (callable): Objective function to minimize
        lower_bound (float or array): Lower bounds for each dimension
        upper_bound (float or array): Upper bounds for each dimension
        dim (int): Problem dimension
        max_iter (int): Maximum number of iterations
        step_size (float): Step size for neighbor generation
        initial_temp (float): Initial temperature
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
        self.step_size = step_size
        self.initial_temp = initial_temp
        self.verbose = verbose

        self.best_solution = None
        self.best_fitness = None
        self.history = []

    def generate_neighbor(self, solution):
        """Generate neighboring solution by adding random perturbation.

        Parameters:
        solution (np.ndarray): Current solution vector

        Returns:
        np.ndarray: Neighbor solution
        """
        perturbation = np.random.uniform(
            -self.step_size,
            self.step_size,
            size=self.dim
        )
        neighbor = np.clip(
            solution + perturbation,
            self.lower_bound,
            self.upper_bound
        )
        return neighbor

    def run(self):
        """Execute Simulated Annealing optimization.

        Parameters:
        None

        Returns:
        tuple: (best_solution, best_fitness, history) where history is list of best fitness per iteration
        """
        if self.verbose:
            print("\n===== Start SA =====")
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

            candidate = self.generate_neighbor(current_solution)
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

            if self.verbose and (i % 10 == 0 or i == self.max_iter):
                print(f"Iteration {i}/{self.max_iter}: best fitness = {self.best_fitness:.6f}")

        if self.verbose:
            print("--- Optimization Results (SA) ---")
            print(f"Best Fitness: {self.best_fitness:.6f}")
            print(f"Best Solution: {self.best_solution}")

        return self.best_solution, self.best_fitness, self.history


class SimulatedAnnealingKnapsack:
    """Simulated Annealing algorithm for knapsack problem."""
    
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
        """Initialize Simulated Annealing for knapsack problem.

        Parameters:
        weights (np.ndarray): Item weights
        values (np.ndarray): Item values
        capacity (float): Maximum weight capacity
        dim (int, optional): Number of items, defaults to len(weights)
        max_iter (int): Maximum number of iterations
        step_size (float): Not used for knapsack (kept for compatibility)
        initial_temp (float): Initial temperature
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
        self.step_size = step_size
        self.initial_temp = initial_temp
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

    def generate_neighbor(self, solution):
        """Generate neighboring solution by flipping one bit (knapsack version).

        Parameters:
        solution (np.ndarray): Current binary solution vector

        Returns:
        np.ndarray: Neighbor solution (repaired if invalid)
        """
        neighbor = solution.copy()
        i = np.random.randint(0, self.dim)
        neighbor[i] = 1 - neighbor[i]
        if not self.is_valid(neighbor):
            neighbor = self.repair(neighbor)
        return neighbor

    def run(self):
        """Execute Simulated Annealing for knapsack problem.

        Parameters:
        None

        Returns:
        tuple: (best_solution, best_fitness, history) where history is list of best fitness per iteration
        """
        if self.verbose:
            print("\n===== Start SA Knapsack =====")
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

            if self.verbose and (iteration % 10 == 0 or iteration == self.max_iter):
                print(f"Iteration {iteration}/{self.max_iter}: best fitness = {self.best_fitness:.2f}")

        if self.verbose:
            print("--- Optimization Results (SA Knapsack) ---")
            print(f"Best Fitness: {self.best_fitness:.2f}")
            print(f"Best Solution: {self.best_solution}")

        return self.best_solution, self.best_fitness, self.history
