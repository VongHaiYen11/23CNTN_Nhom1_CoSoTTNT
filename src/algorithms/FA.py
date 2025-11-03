import numpy as np

def firefly_optimize(
    objective_function,
    lower_bound=-5.12,
    upper_bound=5.12,
    dimension=30,
    population_size=30,
    max_iterations=100,
    alpha=0.5,      # mức độ ngẫu nhiên
    beta0=1.0,      # độ hấp dẫn khi khoảng cách = 0
    gamma=0.5,      # hệ số hấp thụ ánh sáng
    seed=None
):

    if seed is not None:
        np.random.seed(seed)

    # === 1 Khởi tạo quần thể ===
    fireflies = lower_bound + (upper_bound - lower_bound) * np.random.rand(population_size, dimension)
    intensity = np.apply_along_axis(objective_function, 1, fireflies)

    # Lưu nghiệm tốt nhất ban đầu
    best_idx = np.argmin(intensity)
    best_solution = np.copy(fireflies[best_idx])
    best_fitness = intensity[best_idx]
    fitness_history = [best_fitness]

    # === 2 Vòng lặp chính ===
    for iteration in range(max_iterations):
        alpha *= 0.97  # giảm dần mức độ ngẫu nhiên theo iteration

        for i in range(population_size):
            for j in range(population_size):
                if intensity[j] < intensity[i]:  # j sáng hơn i
                    # Tính khoảng cách bình phương
                    r2 = np.sum((fireflies[i] - fireflies[j]) ** 2)

                    # Tính độ hấp dẫn beta theo khoảng cách
                    beta = beta0 * np.exp(-gamma * r2)

                    # Cập nhật vị trí
                    random_step = alpha * (np.random.uniform(-0.5, 0.5, dimension)) * (upper_bound - lower_bound)
                    fireflies[i] += beta * (fireflies[j] - fireflies[i]) + random_step

                    # Giữ trong biên
                    fireflies[i] = np.clip(fireflies[i], lower_bound, upper_bound)

                    # Cập nhật độ sáng mới
                    intensity[i] = objective_function(fireflies[i])

        # Cập nhật nghiệm tốt nhất toàn cục
        best_idx = np.argmin(intensity)
        best_solution = np.copy(fireflies[best_idx])
        best_fitness = intensity[best_idx]
        fitness_history.append(best_fitness)

        if (iteration + 1) % 10 == 0 or iteration == 0:
            print(f"Iteration {iteration+1}/{max_iterations}, Best Fitness = {best_fitness:.6f}")

    # === 3 Trả kết quả ===
    print("\n--- Optimization Results (FA) ---")
    print(f"Best Fitness: {best_fitness}")
    print(f"Best Solution: {best_solution}")

    return best_solution, best_fitness, fitness_history

# import numpy as np

# def firefly_optimize(
#     objective_function,
#     lower_bound=-5.12,
#     upper_bound=5.12,
#     dimension=30,
#     population_size=50,
#     max_iterations=100,
#     alpha=1.0,   # attractiveness at r = 0
#     beta=0.05,   # randomization
#     gamma=1,   # light absorption coefficient
#     seed=None
# ):

#     if seed is not None:
#         np.random.seed(seed)

#     # === 1. Khởi tạo quần thể === #
#     population = lower_bound + (upper_bound - lower_bound) * np.random.rand(population_size, dimension)
#     fitness = np.apply_along_axis(objective_function, 1, population)

#     # Ghi nhận con sáng nhất ban đầu
#     best_idx = np.argmin(fitness)
#     best_solution = np.copy(population[best_idx])
#     best_fitness = fitness[best_idx]
#     fitness_history = [best_fitness]

#     # === 2. Vòng lặp chính === #
#     for iteration in range(max_iterations):

#         for i in range(population_size):
#             for j in range(population_size):
#                 if fitness[j] < fitness[i]:
#                   distance = np.linalg.norm(population[i] - population[j])
#                   attractiveness = np.exp(-gamma * distance**2)
#                   population[i] += beta * attractiveness * (population[j] - population[i]) + alpha * (np.random.rand(dimension) - 0.5)

#             # Giữ trong phạm vi giới hạn
#             population[i] = np.maximum(lower_bound, population[i])
#             population[i] = np.minimum(upper_bound, population[i])

#         # Tính lại độ sáng (fitness)
#         fitness = np.apply_along_axis(objective_function, 1, population)

#         # Cập nhật nghiệm tốt nhất
#         best_index = np.argmin(fitness)
#         best_solution = population[best_index]
#         best_fitness = fitness[best_index]

#         fitness_history.append(best_fitness)
#         print(f"Iteration {iteration+1}/{max_iterations}, Best Fitness: {best_fitness:.6f}")

#     # === 3. Trả kết quả === #
#     print("\n--- Optimization Results ---")
#     print(f"Best Fitness: {best_fitness}")
#     print(f"Best Solution: {best_solution}")

#     return best_solution, best_fitness, fitness_history