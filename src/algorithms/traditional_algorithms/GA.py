import numpy as np

def genetic_algorithm_optimize(fitness_func, max_iteration, x_min, x_max, dimension, alpha=0.9,
                               error=1e-4, npopulation=20, elitism=2, sigma=0.1,
                               verbose=False, seed=None):
    """
    Thuật toán di truyền (Genetic Algorithm) cho tối ưu hóa liên tục.

    Tham số:
      fitness_func: hàm fitness (càng nhỏ càng tốt)
      max_iteration: số vòng lặp tối đa
      x_min, x_max: biên dưới và trên
      dimension: số chiều
      alpha: trọng số lai
      error: ngưỡng dừng fitness
      npopulation: số cá thể trong quần thể
      elitism: số cá thể tốt nhất giữ nguyên
      sigma: độ lệch chuẩn đột biến Gaussian
      verbose: in tiến trình nếu True
      seed: số seed cho random (None nếu không cố định)

    Trả về:
      best_individual: nghiệm tốt nhất
      best_fitness: giá trị fitness tương ứng
    """
    if seed is not None:
        np.random.seed(seed)

    # --- Khởi tạo population ---
    population = np.random.uniform(x_min, x_max, (npopulation, dimension))

    def sort_population(pop):
        fitness_values = np.array([fitness_func(ind) for ind in pop])
        idx = np.argsort(fitness_values)
        return pop[idx]

    def elitism_selection(pop):
        pop_sorted = sort_population(pop)
        return pop_sorted[:elitism].copy()

    def selection(pop, n_select):
        selection_pool = []
        for _ in range(n_select):
            idx1, idx2 = np.random.randint(0, len(pop), 2)
            if fitness_func(pop[idx1]) < fitness_func(pop[idx2]):
                selection_pool.append(pop[idx1])
            else:
                selection_pool.append(pop[idx2])
        return np.unique(selection_pool, axis=0)

    def crossover(selected, n_crossover):
        crossover_pool = []
        for _ in range(n_crossover):
            dad = selected[np.random.randint(0, len(selected))]
            mom = selected[np.random.randint(0, len(selected))]
            offspring = alpha * dad + (1 - alpha) * mom
            crossover_pool.append(offspring)
        return np.unique(crossover_pool, axis=0)

    def mutation(selected, n_mutation):
        mutation_pool = []
        for _ in range(n_mutation):
            parent = selected[np.random.randint(0, len(selected))]
            offspring = np.clip(parent + sigma * np.random.randn(dimension), x_min, x_max)
            mutation_pool.append(offspring)
        return np.unique(mutation_pool, axis=0)

    # --- Vòng lặp chính ---
    for i in range(max_iteration):
        elites = elitism_selection(population)
        new_population = elites.copy()

        while len(new_population) < npopulation:
            remain = npopulation - len(new_population)
            n_selection = max(int(remain * 0.4), 2)
            n_crossover = max(int(remain * 0.4), 1)
            n_mutation = max(remain - n_selection - n_crossover, 1)

            selected = selection(population, n_selection)
            crossed = crossover(selected, n_crossover)
            mutated = mutation(selected, n_mutation)

            combined = np.concatenate((selected, crossed, mutated), axis=0)
            new_population = np.unique(np.concatenate((new_population, combined), axis=0), axis=0)

        population = new_population[:npopulation]
        population = sort_population(population)

        best_fit = fitness_func(population[0])
        if verbose:
            print(f"Iteration {i + 1}: best fitness = {best_fit:.6f}")
        if best_fit < error:
            break

    best_individual = population[0]
    best_fitness = fitness_func(best_individual)
    return best_individual, best_fitness
