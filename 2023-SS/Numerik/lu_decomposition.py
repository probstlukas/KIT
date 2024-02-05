from scipy import linalg 
import numpy as np

def lu_decomposition(matrix):
    n = matrix.shape[0]
    lower = np.zeros(shape=matrix.shape)
    upper = np.zeros(shape=matrix.shape) 
    for j in range(n):
        lower[j][j] = 1.0
        for i in range(j + 1):
            first_sum = sum(upper[k][j] * lower[i][k] for k in range(i))
            upper[i][j] = matrix[i][j] - first_sum
        for i in range(j, n):
            second_sum = sum(upper[k][j] * lower[i][k] for k in range(j))
            lower[i][j] = (matrix[i][j] - second_sum) / upper[j][j] 
    return lower, upper

def forward_sub(lower, rhs): 
    n = lower.shape[0] 
    solution = np.zeros(n) 
    for i in range(n):
        solution[i] = rhs[i] 
        for j in range(i):
            solution[i] = solution[i] - (lower[i, j] * solution[j]) 
        solution[i] = solution[i] / lower[i, i]
    return solution

def backward_sub(upper, rhs):
    n = upper.shape[0]
    solution = np.zeros(n)
    for i in range(n - 1, -1, -1):
        tmp = rhs[i]
        for j in range(i + 1, n):
            tmp -= upper[i, j] * solution[j] 
        solution[i] = tmp / upper[i, i]
    return solution

def solve_with_lu(matrix, rhs):
    lower, upper = lu_decomposition(matrix) 
    y = forward_sub(lower, rhs)
    return backward_sub(upper, y)
