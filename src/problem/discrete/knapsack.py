import numpy as np

def knapsack_fitness(solution: np.ndarray, weights: np.ndarray, values: np.ndarray, capacity: float) -> float:
    """
    Hàm đánh giá (fitness) cho bài toán ba lô 0/1.
    
    Thông tin:
    - Mỗi phần tử trong `solution` là 0 hoặc 1, thể hiện việc chọn hoặc không chọn vật phẩm.
    - `weights`: mảng chứa trọng lượng của từng vật phẩm.
    - `values`: mảng chứa giá trị của từng vật phẩm.
    - `capacity`: sức chứa tối đa của ba lô.
    
    Mục tiêu:
    - Tối đa hóa tổng giá trị, nhưng tổng trọng lượng không vượt quá capacity.
    """
    # Tính tổng trọng lượng của các vật phẩm được chọn
    total_weight = np.sum(weights * solution)
    
    # Tính tổng giá trị của các vật phẩm được chọn
    total_value = np.sum(values * solution)

    # Nếu vượt quá giới hạn sức chứa → phạt điểm (giảm giá trị)
    if total_weight > capacity:
        # Dạng penalty: trừ bớt theo mức độ vượt quá
        penalty = (total_weight - capacity) * 10  # hệ số phạt = 10
        return total_value - penalty
    else:
        return total_value


def generate_knapsack_problem(n_items: int = 20, seed: int = 42):
    """
    Hàm sinh ngẫu nhiên dữ liệu cho bài toán Knapsack.
    
    Thông tin:
    - n_items: số lượng vật phẩm.
    - seed: giá trị random seed để kết quả tái lập được.
    
    Trả về:
    - weights: trọng lượng từng vật phẩm.
    - values: giá trị từng vật phẩm.
    - capacity: sức chứa tối đa (tính bằng 50% tổng trọng lượng).
    """
    rng = np.random.default_rng(seed)

    # Sinh ngẫu nhiên trọng lượng trong khoảng [1, 20]
    weights = rng.integers(1, 20, size=n_items)
    
    # Sinh ngẫu nhiên giá trị trong khoảng [10, 100]
    values = rng.integers(10, 100, size=n_items)
    
    # Sức chứa ba lô bằng 50% tổng trọng lượng → để đảm bảo có ràng buộc
    capacity = int(0.5 * np.sum(weights))
    
    return weights, values, capacity


# ==== Ví dụ sử dụng ====
if __name__ == "__main__":
    # Tạo dữ liệu bài toán
    weights, values, capacity = generate_knapsack_problem(n_items=10)

    # Giả sử ta chọn ngẫu nhiên một nghiệm 0/1
    solution = np.random.randint(0, 2, size=len(weights))

    # Tính điểm fitness của nghiệm này
    score = knapsack_fitness(solution, weights, values, capacity)

    print("Trọng lượng:", weights)
    print("Giá trị:", values)
    print("Sức chứa:", capacity)
    print("Nghiệm:", solution)
    print("Điểm fitness:", score)
