import numpy as np
from numba import njit, prange

'''MaxCut implementation.

numba==0.61.2
numpy==2.2.5
'''

@njit
def int_to_bitstring(i, n):
    """
    Converts an integer to its binary representation as a bitstring array of size n.

    Parameters:
    - i: Integer.
    - n: Number of bits in the output array.

    Returns:
    - A numpy array of 0s and 1s representing the binary form of `i` over `n` bits.
    """
    if i > 0 and n < np.ceil(np.log2(i)):
        raise ValueError("Need larger n to encode the integer.")
    bits = np.empty(n, dtype=np.uint8)
    for j in range(n):
        bits[n - j - 1] = (i >> j) & 1
    return bits

@njit
def bitstring_to_int(bits):
    """
    Converts a binary bitstring array to its integer representation.

    Parameters:
    - bits: A numpy array of 0s and 1s.

    Returns:
    - An integer corresponding to the binary number represented by `bits`.
    """
    result = 0
    n = bits.size
    for j in range(n):
        result |= bits[n - j - 1] << j
    return result

@njit
def maxcut_cost(x, adj_matrix):
    """
    Computes the MaxCut cost of a specific assignment.

    Parameters:
    - x: Binary vector representing a Cut assignment to the graph.
    - adj_matrix: Array representing the adjacency matrix of a graph.

    Returns:
    - Integer representing the MaxCut cost of assignment `x` for the graph described by `adj_matrix`.
    """
    n = len(x)
    cost = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            if x[i] != x[j]:
                cost += adj_matrix[i, j]
    return -cost

@njit(parallel=True)
def all_maxcut_costs(adj_matrix):
    """
    Computes the MaxCut cost of all possible assignments.

    Parameters:
    - adj_matrix: Array representing the adjacency matrix of a graph.

    Returns:
    - Array of integers containing the MaxCut cost of all assignments for the graph described by `adj_matrix`.
    """
    n = adj_matrix.shape[0]
    total_configs = 1 << n  # 2**n
    results = np.zeros(total_configs)
    for x in prange(total_configs):
        results[x] = maxcut_cost(int_to_bitstring(x,n),adj_matrix)

    return results