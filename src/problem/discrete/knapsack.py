import numpy as np

TEST_CASE_5 = {
    'name': 'test_case_5',
    'weights': np.array([4, 3, 6, 5, 7]),
    'values':  np.array([12, 10, 20, 15, 18]),
    'max_weight': 15,
    'n_items': 5
}

TEST_CASE_10 = {
    'name': 'test_case_10',
    'weights': np.array([10,2,5,12,6,7,11,20,3,9]),
    'values':  np.array([25,8,15,30,5,18,22,40,9,28]),
    'max_weight': 25,
    'n_items': 10
}

TEST_CASE_20 = {
    'name': 'test_case_20',
    'weights': np.array([18,4,3,20,9,30,5,10,12,7, 15,2,25,14,1,19,8,11,22,6]),
    'values':  np.array([5,40,12,55,9,60,17,50,4,33, 20,7,65,14,3,28,45,10,70,25]),
    'max_weight': 50,
    'n_items': 20
}

TEST_CASE_30 = {
    'name': 'test_case_30',
    'weights': np.array([3,4,5,6,7,4,5,6,7,8,3,4,5,6,7,4,5,6,7,8,3,4,5,6,7,4,5,6,7,8]),
    'values':  np.array([6,10,12,15,18,10,14,17,20,22,6,10,12,15,18,10,14,17,20,22,6,10,12,15,18,10,14,17,20,22]),
    'max_weight': 80,
    'n_items': 30
}

TEST_CASES = {
    'test_case_5': TEST_CASE_5,
    'test_case_10': TEST_CASE_10,
    'test_case_20': TEST_CASE_20,
    'test_case_30': TEST_CASE_30
}

# Default test case
WEIGHTS = TEST_CASE_20['weights']
VALUES = TEST_CASE_20['values']
MAX_WEIGHT = TEST_CASE_20['max_weight']
N_ITEMS = TEST_CASE_20['n_items']
