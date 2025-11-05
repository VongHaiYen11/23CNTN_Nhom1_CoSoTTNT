import numpy as np

def simulated_annealing_optimize(fitness_func, x_min, x_max, dimension, max_iteration=100,
                                 step_size=0.1, initial_temp=100, error=1e-4,
                                 seed=None, verbose=False):
    """

    Mô tả ngắn:
      - Khởi tạo nghiệm ngẫu nhiên
      - Sinh neighbor quanh nghiệm hiện tại
      - Chấp nhận neighbor tốt hơn hoặc kém hơn với xác suất phụ thuộc nhiệt độ
      - Nhiệt độ giảm dần theo lịch trình làm nguội (T = T0 * (1 - iter / max_iter))

    Tham số:
      fitness_func (callable): hàm đánh giá fitness (minimization)
      x_min, x_max (float hoặc array_like): giới hạn dưới và trên cho biến
      dimension (int): số chiều của nghiệm
      max_iteration (int): số vòng lặp tối đa
      step_size (float): tỷ lệ perturbation
      initial_temp (float): nhiệt độ ban đầu
      error (float): ngưỡng dừng theo giá trị fitness
      seed (int): random seed (tùy chọn)
      verbose (bool): in thông tin tiến trình (tùy chọn)

    Trả về:
      current_solution (np.ndarray): nghiệm cuối cùng
      current_fitness (float): giá trị fitness tương ứng
    """
    if seed is not None:
        np.random.seed(seed)

    x_min = np.array(x_min)
    x_max = np.array(x_max)

    # Khởi tạo nghiệm ban đầu
    current_solution = np.random.uniform(x_min, x_max, size=dimension)
    current_fitness = fitness_func(current_solution)

    for iteration in range(1, max_iteration + 1):
        t = initial_temp * (1 - iteration / max_iteration)
        t = max(t, 1e-8)

        # Sinh neighbor
        step = (x_max - x_min) * step_size
        perturbation = np.random.uniform(-step, step, size=dimension)
        candidate = np.clip(current_solution + perturbation, x_min, x_max)
        candidate_fitness = fitness_func(candidate)

        # Quy tắc chấp nhận nghiệm
        if candidate_fitness < current_fitness or np.random.rand() < np.exp(-(candidate_fitness - current_fitness) / t):
            current_solution = candidate
            current_fitness = candidate_fitness

        if verbose:
            print(f"Iteration {iteration}: T={t:.4f}, Fitness={current_fitness:.6f}")

        if current_fitness <= error:
            break

    return current_solution, current_fitness
