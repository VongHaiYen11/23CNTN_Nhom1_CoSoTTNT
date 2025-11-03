import numpy as np

def pso_optimize(
    objective_function,
    dimension,
    lower_bound,
    upper_bound,
    population_size,
    max_iterations,
    seed,
    w=0.7,
    c1=1.5,
    c2=1.5
):
    """
    Particle Swarm Optimization (PSO) tổng quát cho các bài toán liên tục.

    Parameters
    ----------
    objective_function : callable
        Hàm mục tiêu cần tối ưu hóa (minimization).
    dim : int
        Số chiều của không gian tìm kiếm.
    bounds : tuple
        (min, max) cho các giá trị biến.
    n_particles : int
        Số lượng hạt trong đàn.
    n_iterations : int
        Số vòng lặp tối đa.
    w, c1, c2 : float
        Tham số quán tính và hệ số ảnh hưởng cá nhân / bầy đàn.
    seed : int
        Giá trị ngẫu nhiên để tái lập kết quả.

    Returns
    -------
    gbest : np.ndarray
        Vị trí tốt nhất tìm được.
    gbest_value : float
        Giá trị fitness tốt nhất.
    history : list
        Lịch sử giá trị tốt nhất qua các vòng lặp.
    """

    rng = np.random.default_rng(seed)

    # --- Khởi tạo ---
    positions = rng.uniform(lower_bound, upper_bound, (population_size, dimension))
    velocities = np.zeros((population_size, dimension))
    pbest = positions.copy()
    pbest_values = np.array([objective_function(x) for x in positions])
    gbest = positions[np.argmin(pbest_values)]
    gbest_value = np.min(pbest_values)
    history = [gbest_value]

    # --- Vòng lặp chính ---
    for t in range(max_iterations):
        for i in range(population_size):
            fitness = objective_function(positions[i])

            # Cập nhật best cá nhân
            if fitness < pbest_values[i]:
                pbest[i] = positions[i]
                pbest_values[i] = fitness

        # Cập nhật best toàn cục
        best_idx = np.argmin(pbest_values)
        if pbest_values[best_idx] < gbest_value:
            gbest = pbest[best_idx].copy()
            gbest_value = pbest_values[best_idx]

        # Cập nhật vận tốc và vị trí
        r1, r2 = rng.random((population_size, dimension)), rng.random((population_size, dimension))
        velocities = (
            w * velocities
            + c1 * r1 * (pbest - positions)
            + c2 * r2 * (gbest - positions)
        )
        positions += velocities

        # Giữ trong giới hạn
        positions = np.clip(positions, lower_bound, upper_bound)
        history.append(gbest_value)

    return gbest, gbest_value, history