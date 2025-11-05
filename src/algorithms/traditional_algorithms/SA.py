import numpy as np

def simulated_annealing_optimize(fitness_func, x_min, x_max, dimension, max_iteration=100,
                                 step_size=0.1, initial_temp=100, error=1e-4, seed=None, verbose=False):
    """
    Thực hiện thuật toán Simulated Annealing (SA) cho tối ưu hóa liên tục.

    Mô tả ngắn:
      Hàm này tìm nghiệm tối ưu bằng cách:
      - Khởi tạo một nghiệm ngẫu nhiên
      - Sinh một neighbor ngẫu nhiên quanh nghiệm hiện tại
      - Chấp nhận neighbor tốt hơn hoặc neighbor kém hơn với xác suất giảm theo nhiệt độ
      - Nhiệt độ giảm dần theo số vòng lặp (cooling schedule đơn giản: T = T0 / iteration)
      - Quá trình lặp lại cho đến khi đạt ngưỡng lỗi hoặc không cải thiện liên tiếp
      - Cắt giá trị neighbor vào khoảng [x_min, x_max]

    Tham số:
      fitness_func (callable): hàm nhận một nghiệm (mảng 1D, kích thước = dimension) trả về giá trị fitness (minimization)
      x_min, x_max (float hoặc array_like): biên dưới và biên trên cho các biến
      dimension (int): số chiều của nghiệm
      max_iteration (int, tùy chọn): số vòng lặp tối đa (mặc định 100)
      step_size (float, tùy chọn): bước perturbation theo tỷ lệ khoảng giá trị (mặc định 0.1)
      initial_temp (float, tùy chọn): nhiệt độ ban đầu (mặc định 100)
      error (float, tùy chọn): ngưỡng dừng theo giá trị fitness (mặc định 1e-4)
      max_no_improve (int, tùy chọn): số vòng liên tiếp không cải thiện trước khi dừng (mặc định 10)
      seed (int, tùy chọn): seed cho random, để tái lập kết quả (mặc định None)
      verbose (bool, tùy chọn): nếu True in thông tin tiến trình (mặc định True)

    Trả về:
      best_solution (np.ndarray): nghiệm tốt nhất tìm được (mảng kích thước = dimension)
      best_fitness (float): giá trị fitness tương ứng của best_solution
    """
    if seed is not None:
        np.random.seed(seed)

    x_min = np.array(x_min)
    x_max = np.array(x_max)

    # Khởi tạo nghiệm ban đầu
    best_solution = np.random.uniform(x_min, x_max, size=dimension)
    best_fitness = fitness_func(best_solution)
    current_solution = best_solution.copy()
    current_fitness = best_fitness
    count_not_improve = 0

    for iteration in range(1, max_iteration + 1):
        t = initial_temp * (1 - iteration / max_iteration)
        t = max(t, 1e-8) 
        # Sinh neighbor
        step = (x_max - x_min) * step_size
        perturbation = np.random.uniform(-step, step, size=dimension)
        candidate = np.clip(current_solution + perturbation, x_min, x_max)
        candidate_fitness = fitness_func(candidate)

        # Chấp nhận neighbor
        if candidate_fitness < current_fitness or np.random.rand() < np.exp(-(candidate_fitness - current_fitness) / t):
            current_solution = candidate
            current_fitness = candidate_fitness
            if current_fitness < best_fitness:
                best_solution = current_solution.copy()
                best_fitness = current_fitness

        if verbose:
            print(f"Iteration {iteration}: Temperature={t:.3f}, Best fitness={best_fitness:.6f}")

        if best_fitness <= error:
            break

    return best_solution, best_fitness
