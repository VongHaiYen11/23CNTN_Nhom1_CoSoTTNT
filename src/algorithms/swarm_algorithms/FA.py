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

        self.fireflies = fireflies
        print("\n--- Optimization Results (FA) ---")
        # print(f"Best Fitness: {best_fitness}")
        # print(f"Best Solution: {best_solution}")

        return best_solution, best_fitness, fitness_history
    
#########################################
# Firefly Algorithm cho Knapsack Problem
#########################################

class FireflyKnapsack:
    def __init__(
        self,
        weights,
        values,
        max_weight,
        population_size=100,
        max_iterations=200,
        alpha=0.2,      # Xác suất mutation
        beta0=1.0,      # Độ hấp dẫn cơ bản
        gamma=0.01,     # Hệ số giảm độ hấp dẫn theo khoảng cách Hamming
        seed=None
    ):
        """
        Firefly Algorithm cho 0/1 Knapsack (discrete).
        
        :param weights: List/array trọng lượng các item
        :param values: List/array giá trị các item
        :param max_weight: Trọng lượng tối đa của ba lô
        :param population_size: Số đom đóm
        :param max_iterations: Số vòng lặp
        :param alpha: Tỷ lệ mutation
        :param beta0: Độ hấp dẫn cơ bản
        :param gamma: Hệ số hấp thụ (dựa trên khoảng cách Hamming)
        :param seed: Reproducibility
        """
        self.weights = np.array(weights)
        self.values = np.array(values)
        self.max_weight = max_weight
        self.n_items = len(weights)
        self.population_size = population_size
        self.max_iterations = max_iterations
        self.alpha = alpha
        self.beta0 = beta0
        self.gamma = gamma
        
        self.fireflies = None  # binary solutions
        self.fitness = None    # tổng giá trị (maximize)
        
        if seed is not None:
            np.random.seed(seed)

    def _initialize_population(self):
        """
        Khởi tạo ngẫu nhiên các giải pháp binary (0/1)
        """
        fireflies = np.random.randint(0, 2, size=(self.population_size, self.n_items))
        fitness = np.array([self._evaluate(sol) for sol in fireflies])
        
        best_idx = np.argmax(fitness)
        best_solution = fireflies[best_idx].copy()
        best_fitness = fitness[best_idx]
        fitness_history = [best_fitness]
        
        self.fireflies = fireflies
        self.fitness = fitness
        
        return fireflies, fitness, best_solution, best_fitness, fitness_history

    def _evaluate(self, solution):
        """
        Đánh giá fitness: tổng giá trị, phạt nếu vượt trọng lượng
        """
        total_value = np.sum(solution * self.values)
        total_weight = np.sum(solution * self.weights)
        
        if total_weight > self.max_weight:
            return 0  # phạt nặng: không hợp lệ
        return total_value  # maximize

    def _hamming_distance(self, x, y):
        """Khoảng cách Hamming giữa 2 binary vector"""
        return np.sum(x != y)

    def _crossover(self, parent1, parent2):
        """One-point crossover"""
        if np.random.rand() < 0.8:  # 80% chance crossover
            point = np.random.randint(1, self.n_items)
            child = np.concatenate([parent1[:point], parent2[point:]])
        else:
            child = parent1.copy()
        return child

    def _mutate(self, solution):
        """Bit-flip mutation"""
        for i in range(self.n_items):
            if np.random.rand() < self.alpha:
                solution[i] = 1 - solution[i]  # flip 0 ↔ 1
        return solution

    def _repair(self, solution):
        """
        Sửa giải pháp nếu vượt trọng lượng (greedy drop)
        """
        total_weight = np.sum(solution * self.weights)
        if total_weight <= self.max_weight:
            return solution
        
        # Xóa item có value/weight thấp nhất
        ratios = self.values / (self.weights + 1e-8)
        item_ranks = np.argsort(ratios)
        sol = solution.copy()
        for idx in item_ranks:
            if sol[idx] == 1:
                sol[idx] = 0
                total_weight -= self.weights[idx]
                if total_weight <= self.max_weight:
                    break
        return sol

    def _move_firefly(self, i, j):
        """
        Đom đóm i di chuyển về j (nếu j tốt hơn)
        """
        if self.fitness[j] <= self.fitness[i]:
            return  # không di chuyển
        
        # Tính độ hấp dẫn dựa trên Hamming distance
        r = self._hamming_distance(self.fireflies[i], self.fireflies[j])
        beta = self.beta0 * np.exp(-self.gamma * r)
        
        if np.random.rand() < beta:
            # Crossover + Mutation
            child = self._crossover(self.fireflies[i], self.fireflies[j])
            child = self._mutate(child)
            child = self._repair(child)
            
            # Cập nhật nếu tốt hơn
            child_fitness = self._evaluate(child)
            if child_fitness > self.fitness[i]:
                self.fireflies[i] = child
                self.fitness[i] = child_fitness

    def run(self):
        """
        Chạy FA cho Knapsack
        """
        fireflies, fitness, best_solution, best_fitness, fitness_history = self._initialize_population()
        alpha = self.alpha
        
        for iteration in range(self.max_iterations):
            alpha *= 0.97  # giảm dần mutation
            
            # Mỗi đom đóm xem tất cả các đom đóm khác
            for i in range(self.population_size):
                for j in range(self.population_size):
                    if i != j:
                        self._move_firefly(i, j)
            
            # Cập nhật best
            current_best_idx = np.argmax(fitness)
            if fitness[current_best_idx] > best_fitness:
                best_fitness = fitness[current_best_idx]
                best_solution = fireflies[current_best_idx].copy()
            
            fitness_history.append(best_fitness)
            
            # if (iteration + 1) % 50 == 0:
            #     print(f"Iter {iteration+1}, Best Value: {best_fitness}")

        print("\n--- Optimization Results (Firefly Knapsack) ---")
        total_value = int(best_fitness)
        total_weight = np.sum(best_solution * self.weights)
        print(f"Best Value: {total_value}")
        print(f"Total Weight: {total_weight} <= {self.max_weight}")
        print(f"Selected Items: {best_solution}")
        
        return best_solution, total_value, fitness_history