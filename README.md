# Đồ án 1: Thuật toán tìm kiếm  
**Môn: Cơ sở trí tuệ nhân tạo - Lớp 23TN**  
**Khoa Công nghệ thông tin - Trường Đại học Khoa học Tự nhiên, ĐHQG-HCM**

## 1. Giới thiệu

Dự án triển khai và so sánh các thuật toán tối ưu hóa dựa trên **Trí tuệ Bầy đàn (Swarm Intelligence)** và một số thuật toán truyền thống, áp dụng cho hai bài toán điển hình:

- Bài toán liên tục: **Sphere Function**  
- Bài toán rời rạc: **0/1 Knapsack Problem**

## 2. Các thuật toán đã triển khai

### Swarm Intelligence
- Ant Colony Optimization (ACO)  
- Particle Swarm Optimization (PSO)  
- Artificial Bee Colony (ABC)  
- Firefly Algorithm (FA)  
- Cuckoo Search (CS)

### Thuật toán truyền thống
- Genetic Algorithm (GA)  
- Simulated Annealing (SA)  
- Hill Climbing (HC)

**Báo cáo đầy đủ**:
## 3. Cấu trúc thư mục
.

├── results/ 

├── src/

│   ├── algorithms/

│   │   ├── swarm_algorithms/   

│   │   └── traditional_algorithms/ 

├── experiment/

│   ├── run_sphere.py   

│   └── run_knapsack.py 

├── problem/

│   ├── continuous/ 

│   └── discrete/

└── visualization/

│   ├── parameter_analysis/ 

│   └── performentce/

## 4. Cài đặt thư viện
### Thư viện cho các thuật toán:
```
import numpy as np
```
### Thư viện cho experiment và visualization:
```
import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import random
import importlib.util
```

## 5. Cài đặt và chạy thuật toán:
```
git clone https://github.com/VongHaiYen11/23CNTN_Nhom1_CoSoTTNT.git
cd 23CNTN_Nhom1_CoSoTTNT
```

### Chạy thí nghiệm Sphere Function:
```
python src/experiment/run_sphere.py
```
### Chạy thí nghiệm Knapsack Function:
```
python src/experiment/run_knapsack.py
```
Kết quả sẽ tự động lưu vào thư mục ```results/```.

### Visualization:
```
python src/visualization/performance/visualize_experiments.py
```
Kết quả sẽ là toàn bộ biểu đồ so sánh fitness, thời gian, memory các thuật toán.
### Parameter Analysis:
```
python src/visualization/parameter_analysis/parameter.py
```
Kết quả là biểu đồ so sánh fitness khi thay đổi tham số.

## Thông tin tác giả:
| Thành viên                  | MSSV      |
|-----------------------------|-----------|
| Ung Dung Thanh Hạ           | 23120039  |
| Vòng Hải Yến                | 23120108  |
| Nguyễn Ngọc Duy Mỹ          | 23120145  |

**Giáo viên hướng dẫn**  
- GS.TS. Lê Hoài Bắc  
- Thầy Nguyễn Thanh Tình


*Tháng 11/2025 - Nhóm sinh viên 23TN, Khoa CNTT, ĐH KHTN - ĐHQG HCM*
