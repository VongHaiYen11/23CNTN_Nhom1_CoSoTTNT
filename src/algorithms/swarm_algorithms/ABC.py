import numpy as np

def abc_optimize(
    fitness_function,
    lower_bound=-5.12,
    upper_bound=5.12,
    problem_size=30,
    num_employed_bees=100,
    num_onlooker_bees=100,
    max_iterations=200,
    limit=50,
    seed=None
):

    if seed is not None:
        np.random.seed(seed)

    # === 1. Khởi tạo quần thể === #
    food_sources = lower_bound + np.random.rand(num_employed_bees, problem_size) * (upper_bound - lower_bound)
    fitness_values = np.array([fitness_function(food) for food in food_sources])
    no_improvement_counters = np.zeros(num_employed_bees)

    # Lưu nghiệm tốt nhất
    best_index = np.argmin(fitness_values)
    best_solution = np.copy(food_sources[best_index])
    best_fitness = fitness_values[best_index]
    fitness_history = [best_fitness]

    # === 2. Vòng lặp chính === #
    for iteration in range(max_iterations):

        # ---------- Pha Employed Bees ---------- #
        for i in range(num_employed_bees):
            k = np.random.randint(0, num_employed_bees)
            while k == i:
                k = np.random.randint(0, num_employed_bees)

            dimension = np.random.randint(0, problem_size)
            phi = np.random.uniform(-1, 1)
            mutant = np.copy(food_sources[i])

            mutant[dimension] = (
                food_sources[i][dimension]
                + phi * (food_sources[i][dimension] - food_sources[k][dimension])
            )

            # Ràng buộc biên
            mutant[dimension] = np.clip(mutant[dimension], lower_bound, upper_bound)

            mutant_fitness = fitness_function(mutant)

            if mutant_fitness < fitness_values[i]:
                food_sources[i] = mutant
                fitness_values[i] = mutant_fitness
                no_improvement_counters[i] = 0
            else:
                no_improvement_counters[i] += 1

        # ---------- Tính xác suất cho Onlooker ---------- #
        inv_fit = 1.0 / (1.0 + fitness_values)
        probabilities = inv_fit / np.sum(inv_fit)

        # ---------- Pha Onlooker Bees ---------- #
        cumulative_probs = np.cumsum(probabilities)
        for j in range(num_onlooker_bees):
            # i = np.random.choice(num_employed_bees, p=probabilities)
            r = np.random.rand()
            # Tìm kiếm nhị phân (rất nhanh), chi phí O(log N_E)
            i = np.searchsorted(cumulative_probs, r)
           
            k = np.random.randint(0, num_employed_bees)
            while k == i:
                k = np.random.randint(0, num_employed_bees)

            dimension = np.random.randint(0, problem_size)
            phi = np.random.uniform(-1, 1)
            mutant = np.copy(food_sources[i])

            mutant[dimension] = (
                food_sources[i][dimension]
                + phi * (food_sources[i][dimension] - food_sources[k][dimension])
            )

            mutant[dimension] = np.clip(mutant[dimension], lower_bound, upper_bound)
            mutant_fitness = fitness_function(mutant)

            if mutant_fitness < fitness_values[i]:
                food_sources[i] = mutant
                fitness_values[i] = mutant_fitness
                no_improvement_counters[i] = 0
            else:
                no_improvement_counters[i] += 1

        # ---------- Pha Scout Bees ---------- #
        for i in range(num_employed_bees):
            if no_improvement_counters[i] > limit:
                # tái khởi tạo trong domain gốc
                food_sources[i] = lower_bound + np.random.rand(problem_size) * (upper_bound - lower_bound)
                fitness_values[i] = fitness_function(food_sources[i])
                no_improvement_counters[i] = 0

        # ---------- Cập nhật nghiệm tốt nhất ---------- #
        current_best_index = np.argmin(fitness_values)
        current_best_fitness = fitness_values[current_best_index]

        if current_best_fitness < best_fitness:
            best_fitness = current_best_fitness
            best_solution = np.copy(food_sources[current_best_index])

        fitness_history.append(best_fitness) # type: ignore

        # print(f"Iteration {iteration+1}/{max_iterations}, Best Fitness: {best_fitness:.6f}")

    # === 3. Trả kết quả === #
    print("\n--- Optimization Results ---")
    print(f"Best Fitness: {best_fitness}")
    print(f"Best Solution: {best_solution}")

    return best_solution, best_fitness, fitness_history