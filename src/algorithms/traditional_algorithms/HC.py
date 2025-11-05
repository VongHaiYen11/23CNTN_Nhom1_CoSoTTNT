import numpy as np

def hill_climbing_optimize(fitness_func, x_min, x_max, dimension, max_iteration=100,
                           n_neighbors=1000, error=1e-4, seed=None, verbose=False):
    """
    Thực hiện thuật toán Hill Climbing (tối ưu cục bộ) cho bài toán tối ưu hóa liên tục.

    Mô tả ngắn:
      Hàm này tìm nghiệm tối ưu bằng cách:
      - Khởi tạo một nghiệm ngẫu nhiên
      - Sinh một số neighbors quanh nghiệm hiện tại
      - Chọn neighbor tốt nhất để cập nhật
      - Lặp lại quá trình cho đến khi đạt ngưỡng lỗi hoặc không cải thiện liên tiếp
      - Sử dụng clipping để đảm bảo nghiệm nằm trong khoảng [x_min, x_max]

    Tham số:
      fitness_func (callable): hàm nhận một nghiệm (mảng 1D, kích thước = dimension) trả về giá trị fitness (minimization)
      x_min, x_max (float hoặc array_like): biên dưới và biên trên cho các biến
      dimension (int): số chiều của nghiệm
      max_iteration (int, tùy chọn): số vòng lặp tối đa (mặc định 100)
      n_neighbors (int, tùy chọn): số neighbor được sinh ra mỗi vòng lặp (mặc định 20)
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
    current_solution = np.random.uniform(x_min, x_max, size=dimension)
    current_fitness = fitness_func(current_solution)

    for iteration in range(max_iteration):
        # Sinh neighbors quanh nghiệm hiện tại
        step = (x_max - x_min) * 0.1
        neighbors = []
        for _ in range(n_neighbors):
            perturbation = np.random.uniform(-step, step, size=dimension)
            neighbor = np.clip(current_solution + perturbation, x_min, x_max)
            neighbors.append(neighbor)
        neighbors = np.array(neighbors)

        # Tính fitness và sắp xếp neighbors theo giá trị fitness
        fitness_values = np.array([fitness_func(n) for n in neighbors])
        idx = np.argsort(fitness_values)
        neighbors = neighbors[idx]

        # Cập nhật nghiệm tốt nhất nếu có cải thiện
        if fitness_func(neighbors[0]) < current_fitness:
            current_solution = neighbors[0]
            current_fitness = fitness_func(neighbors[0])
        else:
            break

        if verbose:
            print(f"Iteration {iteration + 1}: best fitness = {current_fitness:.6f}")

        if current_fitness < error:
            break

    return current_solution, current_fitness
