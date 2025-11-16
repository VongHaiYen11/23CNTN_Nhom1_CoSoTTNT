import numpy as np

# Test case 1: 5 items
TEST_CASE_5 = {
    'name': 'test_case_5',
    'weights': np.array([4, 7, 2, 9, 5]),
    'values':  np.array([10, 15, 8, 20, 12]),
    'max_weight': 15,
    'n_items': 5
}

# Test case 2: 10 items
TEST_CASE_10 = {
    'name': 'test_case_10',
    'weights': np.array([3, 6, 2, 5, 7, 4, 8, 3, 6, 5]),
    'values':  np.array([12, 18, 10, 15, 20, 13, 22, 11, 17, 14]),
    'max_weight': 30,
    'n_items': 10
}

# Test case 3: 20 items
TEST_CASE_20 = {
    'name': 'test_case_20',
    'weights': np.array([2,3,4,5,6,3,4,5,6,7,2,3,4,5,6,3,4,5,6,7]),
    'values':  np.array([5,8,12,15,18,7,10,14,17,20,5,8,12,15,18,7,10,14,17,20]),
    'max_weight': 50,
    'n_items': 20
}

# Test case 4: 30 items
TEST_CASE_30 = {
    'name': 'test_case_30',
    'weights': np.array([3,4,5,6,7,4,5,6,7,8,3,4,5,6,7,4,5,6,7,8,3,4,5,6,7,4,5,6,7,8]),
    'values':  np.array([6,10,12,15,18,10,14,17,20,22,6,10,12,15,18,10,14,17,20,22,6,10,12,15,18,10,14,17,20,22]),
    'max_weight': 80,
    'n_items': 30
}

# Combine all test cases
TEST_CASES = {
    'test_case_5': TEST_CASE_5,
    'test_case_10': TEST_CASE_10,
    'test_case_20': TEST_CASE_20,
    'test_case_30': TEST_CASE_30
}

# Default test case
WEIGHTS = TEST_CASE_5['weights']
VALUES = TEST_CASE_5['values']
MAX_WEIGHT = TEST_CASE_5['max_weight']
N_ITEMS = TEST_CASE_5['n_items']
