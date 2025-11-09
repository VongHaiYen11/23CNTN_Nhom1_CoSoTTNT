import numpy as np

class ArtificialBeeColony:
    def __init__(
        self,
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
        """
        Khởi tạo class ArtificialBeeColony với các tham số cần thiết.
        
        :param fitness_function: Hàm tính fitness (minimization).
        :param lower_bound: Giới hạn dưới của không gian tìm kiếm.
        :param upper_bound: Giới hạn trên của không gian tìm kiếm.
        :param problem_size: Kích thước vấn đề (số chiều).
        :param num_employed_bees: Số lượng ong thợ (employed bees).
        :param num_onlooker_bees: Số lượng ong quan sát (onlooker bees).
        :param max_iterations: Số lượng vòng lặp tối đa.
        :param limit: Giới hạn không cải thiện để scout bee tái khởi tạo.
        :param seed: Seed cho random để reproducible.
        """
        if seed is not None:
            np.random.seed(seed)

        self.fitness_function = fitness_function
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.problem_size = problem_size
        self.num_employed_bees = num_employed_bees
        self.num_onlooker_bees = num_onlooker_bees
        self.max_iterations = max_iterations
        self.limit = limit

        self.food_sources = None

    def _initialize_population(self):
        """
        Khởi tạo quần thể nguồn thức ăn (food sources) ban đầu.
        
        :return: food_sources, fitness_values, no_improvement_counters, best_solution, best_fitness, fitness_history
        """
        food_sources = self.lower_bound + np.random.rand(self.num_employed_bees, self.problem_size) * (self.upper_bound - self.lower_bound)
        fitness_values = np.array([self.fitness_function(food) for food in food_sources])
        no_improvement_counters = np.zeros(self.num_employed_bees)

        # Lưu nghiệm tốt nhất ban đầu
        best_index = np.argmin(fitness_values)
        best_solution = np.copy(food_sources[best_index])
        best_fitness = fitness_values[best_index]
        fitness_history = [best_fitness]

        self.food_sources = food_sources
        return food_sources, fitness_values, no_improvement_counters, best_solution, best_fitness, fitness_history

    def _employed_bees_phase(self, food_sources, fitness_values, no_improvement_counters):
        """
        Pha Employed Bees: Mỗi ong thợ tạo mutant và cập nhật nếu tốt hơn.
        
        :param food_sources: Ma trận nguồn thức ăn.
        :param fitness_values: Mảng fitness tương ứng.
        :param no_improvement_counters: Mảng đếm không cải thiện.
        """
        for i in range(self.num_employed_bees):
            k = np.random.randint(0, self.num_employed_bees)
            while k == i:
                k = np.random.randint(0, self.num_employed_bees)

            dimension = np.random.randint(0, self.problem_size)
            phi = np.random.uniform(-1, 1)
            mutant = np.copy(food_sources[i])

            mutant[dimension] = (
                food_sources[i][dimension]
                + phi * (food_sources[i][dimension] - food_sources[k][dimension])
            )

            # Ràng buộc biên
            mutant[dimension] = np.clip(mutant[dimension], self.lower_bound, self.upper_bound)

            mutant_fitness = self.fitness_function(mutant)

            if mutant_fitness < fitness_values[i]:
                food_sources[i] = mutant
                fitness_values[i] = mutant_fitness
                no_improvement_counters[i] = 0
            else:
                no_improvement_counters[i] += 1

    def _calculate_probabilities(self, fitness_values):
        """
        Tính xác suất chọn nguồn thức ăn cho onlooker bees.
        
        :param fitness_values: Mảng fitness.
        :return: probabilities, cumulative_probs
        """
        inv_fit = 1.0 / (1.0 + fitness_values)
        probabilities = inv_fit / np.sum(inv_fit)
        cumulative_probs = np.cumsum(probabilities)
        return probabilities, cumulative_probs

    def _onlooker_bees_phase(self, food_sources, fitness_values, no_improvement_counters, cumulative_probs):
        """
        Pha Onlooker Bees: Ong quan sát chọn nguồn và tạo mutant dựa trên xác suất.
        
        :param food_sources: Ma trận nguồn thức ăn.
        :param fitness_values: Mảng fitness.
        :param no_improvement_counters: Mảng đếm không cải thiện.
        :param cumulative_probs: Mảng tích lũy xác suất.
        """
        for j in range(self.num_onlooker_bees):
            r = np.random.rand()
            i = np.searchsorted(cumulative_probs, r)

            k = np.random.randint(0, self.num_employed_bees)
            while k == i:
                k = np.random.randint(0, self.num_employed_bees)

            dimension = np.random.randint(0, self.problem_size)
            phi = np.random.uniform(-1, 1)
            mutant = np.copy(food_sources[i])

            mutant[dimension] = (
                food_sources[i][dimension]
                + phi * (food_sources[i][dimension] - food_sources[k][dimension])
            )

            mutant[dimension] = np.clip(mutant[dimension], self.lower_bound, self.upper_bound)
            mutant_fitness = self.fitness_function(mutant)

            if mutant_fitness < fitness_values[i]:
                food_sources[i] = mutant
                fitness_values[i] = mutant_fitness
                no_improvement_counters[i] = 0
            else:
                no_improvement_counters[i] += 1

    def _scout_bees_phase(self, food_sources, fitness_values, no_improvement_counters):
        """
        Pha Scout Bees: Tái khởi tạo nguồn thức ăn nếu vượt limit không cải thiện.
        
        :param food_sources: Ma trận nguồn thức ăn.
        :param fitness_values: Mảng fitness.
        :param no_improvement_counters: Mảng đếm không cải thiện.
        """
        for i in range(self.num_employed_bees):
            if no_improvement_counters[i] > self.limit:
                food_sources[i] = self.lower_bound + np.random.rand(self.problem_size) * (self.upper_bound - self.lower_bound)
                fitness_values[i] = self.fitness_function(food_sources[i])
                no_improvement_counters[i] = 0

    def _update_best_solution(self, food_sources, fitness_values, best_solution, best_fitness, fitness_history):
        """
        Cập nhật nghiệm tốt nhất toàn cục.
        
        :param food_sources: Ma trận nguồn thức ăn.
        :param fitness_values: Mảng fitness.
        :param best_solution: Nghiệm tốt nhất hiện tại.
        :param best_fitness: Fitness tốt nhất hiện tại.
        :param fitness_history: Lịch sử fitness.
        :return: best_solution, best_fitness, fitness_history (cập nhật)
        """
        current_best_index = np.argmin(fitness_values)
        current_best_fitness = fitness_values[current_best_index]

        if current_best_fitness < best_fitness:
            best_fitness = current_best_fitness
            best_solution = np.copy(food_sources[current_best_index])

        fitness_history.append(best_fitness)
        return best_solution, best_fitness, fitness_history

    def run(self):
        """
        Chạy thuật toán ABC chính: Khởi tạo và lặp qua các pha.
        
        :return: best_solution, best_fitness, fitness_history
        """
        food_sources, fitness_values, no_improvement_counters, best_solution, best_fitness, fitness_history = self._initialize_population()

        for iteration in range(self.max_iterations):
            self._employed_bees_phase(food_sources, fitness_values, no_improvement_counters)

            probabilities, cumulative_probs = self._calculate_probabilities(fitness_values)

            self._onlooker_bees_phase(food_sources, fitness_values, no_improvement_counters, cumulative_probs)

            self._scout_bees_phase(food_sources, fitness_values, no_improvement_counters)

            best_solution, best_fitness, fitness_history = self._update_best_solution(
                food_sources, fitness_values, best_solution, best_fitness, fitness_history
            )

            # print(f"Iteration {iteration+1}/{self.max_iterations}, Best Fitness: {best_fitness:.6f}")

        print("\n--- Optimization Results ABC ---")
        print(f"Best Fitness: {best_fitness}")
        print(f"Best Solution: {best_solution}")

        self.food_sources = food_sources
        return best_solution, best_fitness, fitness_history