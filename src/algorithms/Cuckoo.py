import numpy as np
import math

def cuckoo_optimize(fitness_func, xmin, xmax, dimension=1, population_size=25,
                             alpha=1.5, beta=2, pa=0.25, max_iterations=200, error=1e-4, seed=None):
    
    if seed is not None:
        np.random.seed(seed)
    
    def initialize_population():
        return np.random.uniform(xmin, xmax, size=(population_size, dimension))

    def levy_flight():
        b = beta
        if b <= 1 or b > 3:
            b = 1.5
        sigma_u = (math.gamma(1 + b) * np.sin(np.pi * b / 2) /
                   (math.gamma((1 + b) / 2) * b * 2 ** ((b - 1) / 2))) ** (1 / b)
        u = np.random.normal(0, sigma_u, size=dimension)
        v = np.random.normal(0, 1, size=dimension)
        step = u / (np.abs(v) ** (1 / b))
        return step

    def get_cuckoo_egg(current_solution):
        step = levy_flight()
        new_sol = current_solution + alpha * step
        new_sol = np.clip(new_sol, xmin, xmax)
        return new_sol

    def get_random_nest():
        return np.random.uniform(xmin, xmax, size=dimension)

    # --- Khởi tạo ---
    population = initialize_population()
    t = 0

    while t < max_iterations:
        nest_idx = np.random.randint(0, population_size)
        new_solution = get_cuckoo_egg(population[nest_idx])

        if fitness_func(new_solution) < fitness_func(population[nest_idx]):
            population[nest_idx] = new_solution

        all_fitness = np.array([fitness_func(x) for x in population])
        sorted_indices = np.argsort(all_fitness)
        population = population[sorted_indices]

        best = population[0]
        best_fit = fitness_func(best)

        print(f"Iteration {t+1}: x = {best}, fitness = {best_fit:.6f}\n")

        n_replace = int(pa * population_size)
        for i in range(1, n_replace + 1):
            population[-i] = get_random_nest()

        if best_fit < error:
            break

        t += 1

    best = population[0]
    return best, fitness_func(best)