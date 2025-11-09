# src/problem/discrete/knapsack_adapter.py
import numpy as np

def knapsack_fitness_adapter(real_solution, weights, values, max_weight, use_sigmoid=True, seed=None):
    """
    Adapter: real vector → binary → Knapsack fitness (minimization)
    Dùng được cho ABC, FA, PSO, CS, ACO
    """
    if seed is not None:
        np.random.seed(seed)

    n = len(weights)
    
    if use_sigmoid:
        # Cách tốt nhất: sigmoid + Bernoulli → exploration mượt
        probs = 1 / (1 + np.exp(-real_solution))
        binary_sol = (np.random.rand(n) < probs).astype(int)
    else:
        # Cách đơn giản: threshold
        threshold = 0.0  # vì domain thường [-10,10] → >0 là 1
        binary_sol = (real_solution > threshold).astype(int)
    
    total_weight = np.sum(binary_sol * weights)
    total_value = np.sum(binary_sol * values)
    
    # Penalty nếu vượt trọng lượng
    if total_weight > max_weight:
        penalty = 1e6 * (total_weight - max_weight)
        return -(total_value - penalty)  # minimization
    else:
        return -total_value  # tối ưu giá trị lớn → fitness nhỏ