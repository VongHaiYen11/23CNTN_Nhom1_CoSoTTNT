import numpy as np

def sphere(x: np.ndarray) -> float:
    """
    Hàm mục tiêu Sphere – dùng phổ biến để kiểm tra thuật toán tối ưu liên tục.
    
    Thông tin:
    - Công thức: f(x) = sum(x_i^2)
    - Miền tìm kiếm thường: [-5.12, 5.12]^n
    - Giá trị tối ưu toàn cục: f(x*) = 0 tại x* = [0, 0, ..., 0]
    """
    # np.sum(x ** 2): tính tổng bình phương từng phần tử trong vector x
    return np.sum(x ** 2)


# ==== Giới hạn biên của hàm ====
# Mỗi chiều (dimension) có giá trị trong khoảng [-5.12, 5.12]
# Ở đây ta định nghĩa sẵn để các thuật toán tối ưu có thể import trực tiếp
BOUNDS_SPHERE = [(-5.12, 5.12)] * 30  # 30 chiều là cấu hình mặc định
