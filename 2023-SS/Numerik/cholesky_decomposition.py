from scipy import linalg
import numpy as np

def cholesky_decomposition(matrix):
    n = matrix.shape[0]
    lower = np.zeros(matrix.shape)
    for n in range(1, n):
        v = linalg.solve_triangular(lower[:n, :n], matrix[:n, n], lower=True)
        lower[n, n] = np.sqrt(matrix[n, n] - np.dot(v, v))
    return lower

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

def solve_with_cholesky(matrix, rhs):
    lower = cholesky_decomposition(matrix)
    y = forward_sub(lower, rhs)
    return backward_sub(lower.transpose(), y)
