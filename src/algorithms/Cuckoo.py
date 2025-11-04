import numpy as np
import math

def cs_optimize(fitness_func, xmin, xmax, dim=1,
                         population_size=25, alpha=1, beta=1.5,
                         pa=0.25, max_iter=100, error=1e-4, seed=None):
    if seed is not None:
        np.random.seed(seed)

    # --- Lévy flight generator ---
    def levy_flight(beta):
        if beta <= 1 or beta > 3:
            beta = 1.5
        sigma_u = (math.gamma(1 + beta) * math.sin(math.pi * beta / 2) /
                   (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
        u = np.random.normal(0, sigma_u, size=dim)
        v = np.random.normal(0, 1, size=dim)
        step = u / (np.abs(v) ** (1 / beta))
        return step

    # --- Initialization ---
    nests = np.random.uniform(xmin, xmax, (population_size, dim))
    fitness = np.array([fitness_func(x) for x in nests])

    # --- Main loop ---
    for t in range(max_iter):
        # choose one nest randomly
        i = np.random.randint(0, population_size)

        # sort nests by fitness
        fitness = np.array([fitness_func(x) for x in nests])
        sorted_idx = np.argsort(fitness)
        nests = nests[sorted_idx]
        fitness = fitness[sorted_idx]

        # create a new solution from best
        new_sol = nests[0] + alpha * levy_flight(beta)
        new_sol = np.clip(new_sol, xmin, xmax)

        # greedy replacement
        if fitness_func(new_sol) < fitness_func(nests[i]):
            nests[i] = new_sol

        # abandon fraction pa of worst nests
        n_replace = int(pa * population_size)
        for j in range(1, n_replace + 1):
            nests[-j] = np.random.uniform(xmin, xmax, dim)
            fitness[-j] = fitness_func(nests[-j])

        # check convergence
        if fitness[0] < error:
            break

    best = nests[0]
    return best, fitness_func(best)
