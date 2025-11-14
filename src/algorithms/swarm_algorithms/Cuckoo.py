import numpy as np
import math

class CuckooSearch:
    def __init__(self, fitness_func, lower_bound, upper_bound, dim=1,
                 population_size=25, pa=0.25, alpha=0.01, beta=1.5,
                 max_iter=1000, seed=None, verbose=False):
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
    
    def run(self):
        """Main optimization loop."""
        print("\n--- Cuckoo Search ---")
        n = self.population_size
        hist = []
        nests = np.random.uniform(self.lower_bound, self.upper_bound, (n, self.dim))
        fitness = np.array([self.fitness_func(x) for x in nests])

        best_idx = np.argmin(fitness)
        best = nests[best_idx].copy()
        fmin = fitness[best_idx]
        t = 0
        while t < self.max_iter:
            # Lévy flight phase
            new_nests = self.get_cuckoos(nests, best)
            nests, fitness, best, fmin = self.get_best_nest(nests, new_nests, fitness)

            # Empty nest phase
            new_nests = self.empty_nests(nests)
            nests, fitness, best, fmin = self.get_best_nest(nests, new_nests, fitness)

            hist.append(fmin)
            # if self.verbose and (t % 50 == 0 or t == self.max_iter - 1):
            #     print(f"Iteration {t+1}/{self.max_iter}: best fitness = {fmin:.6f}")


            t += 1
        print("\n--- Optimization Results (Cuckoo Search) ---")
        return best, fmin, hist

##################################################
## Cuckoo Search for Knapsack Problem
##################################################

import numpy as np
import math

class CuckooSearchKnapsack:
    def __init__(self, weights, values, capacity, dim=None,
                 population_size=25, pa=0.25, alpha=0.01, beta=1.5,
                 max_iter=1000, seed=None, verbose=True):
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

    def fitness(self, solution):
        return np.sum(solution * self.values)

    def is_valid(self, solution):
        return np.sum(solution * self.weights) <= self.capacity

    def get_random_nest(self):
        """Sinh 1 nest hợp lệ ngẫu nhiên."""
        while True:
            nest = np.random.randint(0, 2, size=self.dim)
            if self.is_valid(nest):
                return nest

    def initialize_population(self):
        return np.array([self.get_random_nest() for _ in range(self.population_size)])

    def levy_flight(self, size):
        """Lévy flight step using Mantegna’s algorithm."""
        beta = self.beta
        sigma = (math.gamma(1 + beta) * math.sin(math.pi * beta / 2) /
                 (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
        u = np.random.normal(0, sigma, size)
        v = np.random.normal(0, 1, size)
        return u / (np.abs(v) ** (1 / beta))

    def sigmoid_solution(self, x):
        """Map continuous solution to 0/1 and enforce capacity."""
        s = 1 / (1 + np.exp(-x))
        sol = (s > 0.5).astype(int)
        while not self.is_valid(sol):
            ones = np.where(sol == 1)[0]
            sol[np.random.choice(ones)] = 0
        return sol

    def get_best_nest(self, nests, new_nests, fitness):
        """Cập nhật nests nếu new_nests tốt hơn và đảm bảo fitness khớp."""
        new_fitness = np.array([self.fitness(x) for x in new_nests])
        improved = new_fitness > fitness

        for i in range(self.population_size):
            if improved[i]:
                nests[i] = new_nests[i].copy()
                fitness[i] = new_fitness[i]

        best_idx = np.argmax(fitness)
        
        # Debug 
        # for i, nest in enumerate(nests):
        #     print(f"Nest {i}: {nest}, fitness = {self.fitness(nest)}")
        # print("Best index:", best_idx)
        # print("Best nest:", nests[best_idx])
        # print("Best fitness:", fitness[best_idx])

        return nests, fitness, nests[best_idx].copy(), fitness[best_idx]

    def empty_nests(self, nests):
        """Thay thế một số nest với xác suất pa bằng get_random_nest."""
        new_nests = nests.copy()
        for i in range(self.population_size):
            if np.random.rand() < self.pa:
                new_nests[i] = self.get_random_nest()
        return new_nests

    def get_cuckoos(self, nests, best):
        """Sinh nests mới bằng Lévy flight + sigmoid."""
        n = self.population_size
        steps = self.levy_flight((n, self.dim))
        stepsize = self.alpha * steps * (nests - best)
        new_nests = nests + stepsize * np.random.randn(n, self.dim)
        new_nests = np.array([self.sigmoid_solution(x) for x in new_nests])
        return new_nests

    def run(self):
        """Main optimization loop."""
        nests = self.initialize_population()
        fitness = np.array([self.fitness(x) for x in nests])
        best_idx = np.argmax(fitness)
        best = nests[best_idx].copy()
        fmax = fitness[best_idx]

        hist = []
        for t in range(1, self.max_iter + 1):
            # if self.verbose:
            #     print(f"\nIteration {t}/{self.max_iter}")
            #     print("Cuckoo Eggs Phase:")
            
            # Lévy flight phase
            new_nests = self.get_cuckoos(nests, best)
            nests, fitness, best, fmax = self.get_best_nest(nests, new_nests, fitness)

            # if self.verbose:
            #     print("Empty Eggs Phase:")

            # Empty nest phase
            
            new_nests = self.empty_nests(nests)
            nests, fitness, best, fmax = self.get_best_nest(nests, new_nests, fitness)
            hist.append(fmax)

        return best, fmax, hist

if __name__ == "__main__":
    # Ví dụ nhỏ Knapsack
    weights = [2, 3, 4, 5, 9, 7, 1, 6, 8, 3]
    values =  [3, 4, 5, 8, 10, 6, 2, 7, 9, 5]
    capacity = 20

    # Khởi tạo Cuckoo Search cho Knapsack
    cs_knapsack = CuckooSearchKnapsack(
        weights=weights,
        values=values,
        capacity=capacity,
        population_size=5,  # nhỏ để debug dễ
        max_iter=10,        # ít vòng lặp để test nhanh
        alpha=0.01,
        pa=0.25,
        beta=1.5,
        verbose=True,        # in debug mỗi 50 iteration
        seed=42
    )

    best_solution, best_fitness = cs_knapsack.run()

    # In kết quả cuối
    selected_items = np.where(best_solution == 1)[0]
    print("\n=== Final Result ===")
    print(f"Best solution: {best_solution}")
    print(f"Selected items: {selected_items}")
    print(f"Total weight: {np.sum(np.array(weights)[selected_items])}")
    print(f"Total value: {best_fitness}")
