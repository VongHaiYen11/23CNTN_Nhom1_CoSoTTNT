import numpy as np

def rastrigin(x: np.ndarray) -> float:
    """
    Hàm mục tiêu Rastrigin – bài toán kiểm thử nổi tiếng với nhiều cực trị địa phương.
    
    Thông tin:
    - Công thức: f(x) = 10n + sum( x_i^2 - 10*cos(2πx_i) )
    - Miền tìm kiếm thường: [-5.12, 5.12]^n
    - Giá trị tối ưu toàn cục: f(x*) = 0 tại x* = [0, 0, ..., 0]
    """
    # Lấy số chiều của vector đầu vào
    n = len(x)
    
    # Tính từng phần tử theo công thức Rastrigin:
    # 10n: phần hằng cố định (mỗi chiều đóng góp 10)
    # x_i^2: phần phạt theo độ lớn của x
    # -10*cos(2πx_i): tạo ra nhiều điểm cực trị giả
    return 10 * n + np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x))


# ==== Giới hạn biên của hàm ====
# Hàm này cũng được định nghĩa trên khoảng [-5.12, 5.12] cho mỗi chiều
BOUNDS_RASTRIGIN = [(-5.12, 5.12)] * 30
