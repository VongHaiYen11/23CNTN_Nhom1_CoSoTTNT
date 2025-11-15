import numpy as np


class ArtificialBeeColony:
    def __init__(
        self,
        fitness_func,
        lower_bound=-5.12,
        upper_bound=5.12,
        dim=30,
        num_employed_bees=100,
        num_onlooker_bees=100,
        max_iter=200,
        limit=50,
        seed=None,
        verbose=False
    ):
        if seed is not None:
            np.random.seed(seed)

        self.fitness_func = fitness_func
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.dim = dim
        self.num_employed_bees = num_employed_bees
        self.num_onlooker_bees = num_onlooker_bees
        self.max_iter = max_iter
        self.limit = limit
        self.verbose = verbose
        self.food_sources = None

    def initialize_population(self):
        food_sources = (
            self.lower_bound
            + np.random.rand(self.num_employed_bees, self.dim)
            * (self.upper_bound - self.lower_bound)
        )
        fitness_values = np.array([self.fitness_func(food) for food in food_sources])
        no_improvement_counters = np.zeros(self.num_employed_bees)

        best_index = np.argmin(fitness_values)
        best_solution = np.copy(food_sources[best_index])
        best_fitness = fitness_values[best_index]
        fitness_history = [best_fitness]

        self.food_sources = food_sources
        self.best_solution = best_solution
        self.best_fitness = best_fitness
        self.history = fitness_history
        return (
            food_sources,
            fitness_values,
            no_improvement_counters,
            best_solution,
            best_fitness,
            fitness_history
        )

    def employed_bees_phase(self, food_sources, fitness_values, no_improvement_counters):
        for i in range(self.num_employed_bees):
            k = np.random.randint(0, self.num_employed_bees)
            while k == i:
                k = np.random.randint(0, self.num_employed_bees)

            dimension = np.random.randint(0, self.dim)
            phi = np.random.uniform(-1, 1)
            mutant = np.copy(food_sources[i])

            mutant[dimension] = (
                food_sources[i][dimension]
                + phi * (food_sources[i][dimension] - food_sources[k][dimension])
            )

            mutant[dimension] = np.clip(
                mutant[dimension],
                self.lower_bound,
                self.upper_bound
            )
            mutant_fitness = self.fitness_func(mutant)

            if mutant_fitness < fitness_values[i]:
                food_sources[i] = mutant
                fitness_values[i] = mutant_fitness
                no_improvement_counters[i] = 0
            else:
                no_improvement_counters[i] += 1

    def calculate_probabilities(self, fitness_values):
        inv_fit = 1.0 / (1.0 + fitness_values)
        probabilities = inv_fit / np.sum(inv_fit)
        cumulative_probs = np.cumsum(probabilities)
        return probabilities, cumulative_probs

    def onlooker_bees_phase(
        self,
        food_sources,
        fitness_values,
        no_improvement_counters,
        cumulative_probs
    ):
        for j in range(self.num_onlooker_bees):
            r = np.random.rand()
            i = np.searchsorted(cumulative_probs, r)

            k = np.random.randint(0, self.num_employed_bees)
            while k == i:
                k = np.random.randint(0, self.num_employed_bees)

            dimension = np.random.randint(0, self.dim)
            phi = np.random.uniform(-1, 1)
            mutant = np.copy(food_sources[i])

            mutant[dimension] = (
                food_sources[i][dimension]
                + phi * (food_sources[i][dimension] - food_sources[k][dimension])
            )

            mutant[dimension] = np.clip(
                mutant[dimension],
                self.lower_bound,
                self.upper_bound
            )
            mutant_fitness = self.fitness_func(mutant)

            if mutant_fitness < fitness_values[i]:
                food_sources[i] = mutant
                fitness_values[i] = mutant_fitness
                no_improvement_counters[i] = 0
            else:
                no_improvement_counters[i] += 1

    def scout_bees_phase(self, food_sources, fitness_values, no_improvement_counters):
        for i in range(self.num_employed_bees):
            if no_improvement_counters[i] > self.limit:
                food_sources[i] = (
                    self.lower_bound
                    + np.random.rand(self.dim)
                    * (self.upper_bound - self.lower_bound)
                )
                fitness_values[i] = self.fitness_func(food_sources[i])
                no_improvement_counters[i] = 0

    def update_best_solution(
        self,
        food_sources,
        fitness_values,
        best_solution,
        best_fitness,
        fitness_history
    ):
        current_best_index = np.argmin(fitness_values)
        current_best_fitness = fitness_values[current_best_index]

        if current_best_fitness < best_fitness:
            best_fitness = current_best_fitness
            best_solution = np.copy(food_sources[current_best_index])

        fitness_history.append(best_fitness)
        return best_solution, best_fitness, fitness_history

    def run(self):
        (
            food_sources,
            fitness_values,
            no_improvement_counters,
            _,
            _,
            _
        ) = self.initialize_population()

        for iteration in range(self.max_iter):
            self.employed_bees_phase(food_sources, fitness_values, no_improvement_counters)

            probabilities, cumulative_probs = self.calculate_probabilities(fitness_values)

            self.onlooker_bees_phase(
                food_sources,
                fitness_values,
                no_improvement_counters,
                cumulative_probs
            )

            self.scout_bees_phase(food_sources, fitness_values, no_improvement_counters)

            self.best_solution, self.best_fitness, self.history = self.update_best_solution(
                food_sources,
                fitness_values,
                self.best_solution,
                self.best_fitness,
                self.history
            )

            if self.verbose and (iteration % 50 == 0 or iteration == self.max_iter - 1):
                print(f"Iteration {iteration+1}/{self.max_iter}: best fitness = {self.best_fitness:.6f}")

        if self.verbose:
            print("\n--- Optimization Results (ABC) ---")
            print(f"Best Fitness: {self.best_fitness:.6f}")
            print(f"Best Solution: {self.best_solution}")

        self.food_sources = food_sources
        return self.best_solution, self.best_fitness, self.history


class ArtificialBeeColonyKnapsack:
    def __init__(
        self,
        weights,
        values,
        max_weight,
        num_employed_bees=50,
        num_onlooker_bees=50,
        max_iter=200,
        limit=30,
        dim=None,
        seed=None,
        verbose=False
    ):
        if seed is not None:
            np.random.seed(seed)

        self.weights = np.array(weights, dtype=float)
        self.values = np.array(values, dtype=float)
        self.max_weight = float(max_weight)
        self.num_items = len(weights)

        self.num_employed_bees = num_employed_bees
        self.num_onlooker_bees = num_onlooker_bees
        self.max_iter = max_iter
        self.limit = limit
        self.verbose = verbose
        self.food_sources = None

    def initialize_population(self):
        SN = self.num_employed_bees
        food_sources = np.random.randint(0, 2, size=(SN, self.num_items))
        food_sources = np.array([self.repair_solution(ind) for ind in food_sources])

        fitness_values = np.array([self.fitness(ind) for ind in food_sources])
        no_improvement_counters = np.zeros(SN)

        best_idx = np.argmax(fitness_values)
        best_solution = food_sources[best_idx].copy()
        best_fitness = fitness_values[best_idx]
        fitness_history = [best_fitness]

        self.food_sources = food_sources
        self.best_solution = best_solution
        self.best_fitness = best_fitness
        self.history = fitness_history
        return (
            food_sources,
            fitness_values,
            no_improvement_counters,
            best_solution,
            best_fitness,
            fitness_history
        )

    def is_valid(self, solution):
        return np.sum(solution * self.weights) <= self.max_weight

    def repair_solution(self, individual):
        ind = individual.copy()
        while not self.is_valid(ind):
            ones = np.where(ind == 1)[0]
            if len(ones) == 0:
                break
            ind[np.random.choice(ones)] = 0
        return ind

    def fitness(self, individual):
        return np.dot(individual, self.values)

    def binary_mutation(self, x_i, x_k):
        mutant = x_i.copy()
        phi = np.random.uniform(-1, 1, size=self.num_items)
        diff = x_i - x_k
        prob_flip = np.abs(phi) * np.abs(diff)

        flip_mask = np.random.rand(self.num_items) < prob_flip
        mutant[flip_mask] = 1 - mutant[flip_mask]

        return self.repair_solution(mutant)

    def employed_bees_phase(self, food_sources, fitness_values, counters):
        for i in range(self.num_employed_bees):
            k = np.random.randint(0, self.num_employed_bees)
            while k == i:
                k = np.random.randint(0, self.num_employed_bees)

            mutant = self.binary_mutation(food_sources[i], food_sources[k])
            mutant_fitness = self.fitness(mutant)

            if mutant_fitness > fitness_values[i]:
                food_sources[i] = mutant
                fitness_values[i] = mutant_fitness
                counters[i] = 0
            else:
                counters[i] += 1

    def calculate_probabilities(self, fitness_values):
        prob = fitness_values
        prob = np.maximum(prob, 0)
        prob_sum = prob.sum()
        if prob_sum == 0:
            probabilities = np.ones(len(fitness_values)) / len(fitness_values)
        else:
            probabilities = prob / prob_sum
        cumulative = np.cumsum(probabilities)
        return probabilities, cumulative

    def onlooker_bees_phase(self, food_sources, fitness_values, counters, cumulative):
        for _ in range(self.num_onlooker_bees):
            r = np.random.rand()
            i = np.searchsorted(cumulative, r)

            k = np.random.randint(0, self.num_employed_bees)
            while k == i:
                k = np.random.randint(0, self.num_employed_bees)

            mutant = self.binary_mutation(food_sources[i], food_sources[k])
            mutant_fitness = self.fitness(mutant)

            if mutant_fitness > fitness_values[i]:
                food_sources[i] = mutant
                fitness_values[i] = mutant_fitness
                counters[i] = 0
            else:
                counters[i] += 1

    def scout_bees_phase(self, food_sources, fitness_values, counters):
        for i in range(self.num_employed_bees):
            if counters[i] > self.limit:
                food_sources[i] = self.repair_solution(
                    np.random.randint(0, 2, size=self.num_items)
                )
                fitness_values[i] = self.fitness(food_sources[i])
                counters[i] = 0

    def update_best(self, food_sources, fitness_values, best_sol, best_fit, history):
        current_best_idx = np.argmax(fitness_values)
        current_best_fit = fitness_values[current_best_idx]

        if current_best_fit > best_fit:
            best_fit = current_best_fit
            best_sol = food_sources[current_best_idx].copy()

        history.append(best_fit)
        return best_sol, best_fit, history

    def run(self):
        (
            food_sources,
            fitness_values,
            counters,
            _,
            _,
            _
        ) = self.initialize_population()

        for it in range(self.max_iter):
            self.employed_bees_phase(food_sources, fitness_values, counters)
            _, cumulative = self.calculate_probabilities(fitness_values)
            self.onlooker_bees_phase(food_sources, fitness_values, counters, cumulative)
            self.scout_bees_phase(food_sources, fitness_values, counters)
            self.best_solution, self.best_fitness, self.history = self.update_best(
                food_sources,
                fitness_values,
                self.best_solution,
                self.best_fitness,
                self.history
            )

            if self.verbose and (it % 50 == 0 or it == self.max_iter - 1):
                print(f"Iteration {it+1}/{self.max_iter}: best value = {self.best_fitness:.2f}")

        best_items = np.where(self.best_solution == 1)[0]

        if self.verbose:
            print("\n--- Optimization Results (ABC Knapsack) ---")
            print(f"Best Value: {self.best_fitness:.2f}")
            print(f"Selected Items: {best_items.tolist()}")
            print(f"Total Weight: {np.dot(self.best_solution, self.weights):.2f}/{self.max_weight:.2f}")

        return self.best_solution, self.best_fitness, self.history
