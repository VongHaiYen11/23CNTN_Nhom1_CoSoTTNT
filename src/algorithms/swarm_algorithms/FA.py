import numpy as np

class FireflyAlgorithm:
    def __init__(
        self,
        objective_function,
        lower_bound=-5.12,
        upper_bound=5.12,
        dimension=30,
        population_size=100,
        max_iterations=200,
        alpha=0.5,      # Mức độ ngẫu nhiên ban đầu
        beta0=1.0,      # Độ hấp dẫn khi khoảng cách = 0
        gamma=0.01,     # Hệ số hấp thụ ánh sáng
        seed=None
        
    ):
        """
        Khởi tạo class FireflyAlgorithm với các tham số cần thiết.
        
        :param objective_function: Hàm mục tiêu (minimization).
        :param lower_bound: Giới hạn dưới của không gian tìm kiếm.
        :param upper_bound: Giới hạn trên của không gian tìm kiếm.
        :param dimension: Kích thước vấn đề (số chiều).
        :param population_size: Số lượng đom đóm.
        :param max_iterations: Số lượng vòng lặp tối đa.
        :param alpha: Mức độ ngẫu nhiên ban đầu (sẽ giảm dần).
        :param beta0: Độ hấp dẫn cơ bản.
        :param gamma: Hệ số hấp thụ ánh sáng.
        :param seed: Seed cho random để reproducible.
        """

        self.fireflies = None
        if seed is not None:
            np.random.seed(seed)

        self.objective_function = objective_function
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.dimension = dimension
        self.population_size = population_size
        self.max_iterations = max_iterations
        self.alpha = alpha
        self.beta0 = beta0
        self.gamma = gamma

    def _initialize_population(self):
        """
        Khởi tạo quần thể đom đóm ban đầu.
        
        :return: fireflies, intensity, best_solution, best_fitness, fitness_history
        """
        fireflies = self.lower_bound + (self.upper_bound - self.lower_bound) * np.random.rand(self.population_size, self.dimension)
        intensity = np.apply_along_axis(self.objective_function, 1, fireflies)

        # Lưu nghiệm tốt nhất ban đầu
        best_idx = np.argmin(intensity)
        best_solution = np.copy(fireflies[best_idx])
        best_fitness = intensity[best_idx]
        fitness_history = [best_fitness]

        self.fireflies = fireflies
        return fireflies, intensity, best_solution, best_fitness, fitness_history

    def _move_fireflies(self, fireflies, intensity, alpha):
        """
        Pha di chuyển đom đóm: Mỗi đom đóm i di chuyển về j nếu j sáng hơn (intensity thấp hơn).
        
        :param fireflies: Ma trận vị trí đom đóm.
        :param intensity: Mảng độ sáng (fitness).
        :param alpha: Mức độ ngẫu nhiên hiện tại.
        """
        for i in range(self.population_size):
            for j in range(self.population_size):
                if intensity[j] < intensity[i]:  # j sáng hơn i
                    # Tính khoảng cách bình phương
                    r2 = np.sum((fireflies[i] - fireflies[j]) ** 2)

                    # Tính độ hấp dẫn beta theo khoảng cách
                    beta = self.beta0 * np.exp(-self.gamma * r2)

                    # Cập nhật vị trí
                    random_step = alpha * (np.random.uniform(-0.5, 0.5, self.dimension)) * (self.upper_bound - self.lower_bound)
                    fireflies[i] += beta * (fireflies[j] - fireflies[i]) + random_step

                    # Giữ trong biên
                    fireflies[i] = np.clip(fireflies[i], self.lower_bound, self.upper_bound)

                    # Cập nhật độ sáng mới
                    intensity[i] = self.objective_function(fireflies[i])

    def _update_best_solution(self, fireflies, intensity, best_solution, best_fitness, fitness_history):
        """
        Cập nhật nghiệm tốt nhất toàn cục.
        
        :param fireflies: Ma trận vị trí đom đóm.
        :param intensity: Mảng độ sáng (fitness).
        :param best_solution: Nghiệm tốt nhất hiện tại.
        :param best_fitness: Fitness tốt nhất hiện tại.
        :param fitness_history: Lịch sử fitness.
        :return: best_solution, best_fitness, fitness_history (cập nhật)
        """
        current_best_idx = np.argmin(intensity)
        current_best_fitness = intensity[current_best_idx]

        if current_best_fitness < best_fitness:
            best_fitness = current_best_fitness
            best_solution = np.copy(fireflies[current_best_idx])

        fitness_history.append(best_fitness)
        return best_solution, best_fitness, fitness_history

    def run(self):
        """
        Chạy thuật toán FA chính: Khởi tạo và lặp qua các pha di chuyển.
        
        :return: best_solution, best_fitness, fitness_history
        """
        fireflies, intensity, best_solution, best_fitness, fitness_history = self._initialize_population()
        alpha = self.alpha  # Sao chép để giảm dần mà không thay đổi init

        for iteration in range(self.max_iterations):
            alpha *= 0.97  # Giảm dần mức độ ngẫu nhiên theo iteration

            self._move_fireflies(fireflies, intensity, alpha)

            best_solution, best_fitness, fitness_history = self._update_best_solution(
                fireflies, intensity, best_solution, best_fitness, fitness_history
            )

            # if (iteration + 1) % 10 == 0 or iteration == 0:
            #     print(f"Iteration {iteration+1}/{self.max_iterations}, Best Fitness = {best_fitness:.6f}")

        self.fireflies = fireflies
        print("\n--- Optimization Results (FA) ---")
        print(f"Best Fitness: {best_fitness}")
        print(f"Best Solution: {best_solution}")

        return best_solution, best_fitness, fitness_history