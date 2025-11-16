import numpy as np

# Test case 1: Original test case
TEST_CASE_1 = {
    'name': 'test_case_1',
    'weights': np.array([4, 3, 8, 12, 9, 4, 13, 15, 7, 11, 5, 10, 6, 9, 14, 4, 5, 3, 16, 13]),
    'values': np.array([10, 8, 15, 20, 18, 12, 25, 30, 16, 22, 14, 19, 17, 21, 24, 11, 13, 9, 28, 26]),
    'max_weight': 50,
    'n_items': 20
}

# Test case 2: Smaller problem
TEST_CASE_2 = {
    'name': 'test_case_2',
    'weights': np.array([10, 20, 30]),
    'values': np.array([60, 100, 120]),
    'max_weight': 50,
    'n_items': 3
}

# Test case 3: Medium problem with different weight/value ratio
TEST_CASE_3 = {
    'name': 'test_case_3',
    'weights': np.array([2, 3, 4, 5, 6, 7, 8, 9, 10, 11]),
    'values': np.array([5, 8, 12, 15, 18, 22, 25, 28, 30, 35]),
    'max_weight': 30,
    'n_items': 10
}

# Test case 4: Larger problem
TEST_CASE_4 = {
    'name': 'test_case_4',
    'weights': np.array([5, 7, 3, 9, 12, 4, 8, 6, 11, 10, 13, 5, 7, 9, 6, 8, 4, 10, 12, 11, 6, 9, 7, 5, 8]),
    'values': np.array([12, 18, 8, 22, 28, 10, 20, 15, 26, 24, 30, 12, 18, 22, 15, 20, 10, 24, 28, 26, 15, 22, 18, 12, 20]),
    'max_weight': 80,
    'n_items': 25
}

# All test cases
TEST_CASES = {
    'test_case_1': TEST_CASE_1,
    'test_case_2': TEST_CASE_2,
    'test_case_3': TEST_CASE_3,
    'test_case_4': TEST_CASE_4
}

# Default test case (for backward compatibility)
WEIGHTS = TEST_CASE_1['weights']
VALUES = TEST_CASE_1['values']
MAX_WEIGHT = TEST_CASE_1['max_weight']
N_ITEMS = TEST_CASE_1['n_items']