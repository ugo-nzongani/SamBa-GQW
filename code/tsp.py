import numpy as np
import math
from numba import njit, prange

'''
TSP implementation.

numpy==2.2.5
numba==0.61.2
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
def H1(a, b):
    result = 1
    for i in range(len(a)):
        result *= 1 - (a[i] - b[i]) ** 2
    return result

@njit
def H_valid(x_t, tilde_x, K_0):
    n_bits = len(x_t)
    result = 0.0
    for i in K_0:
        term = x_t[i]
        for k in range(i + 1, n_bits):
            term *= 1 - (x_t[k] - tilde_x[k]) ** 2
        result += term
    return result

@njit
def decode_assignment(x, n, n_bits):
    cities = np.zeros((n, n_bits), dtype=np.uint8)
    for i in range(n):
        for j in range(n_bits):
            cities[i, j] = x[i * n_bits + j]
    return cities

def decode_path(vec, n_cities):
    """
    Decodes the TSP encoding to a list of integers representing the cities in their order of visit.

    Parameters:
    - vec: array-like of 0/1 of length n_cities * log2(n_cities)
    - n_cities: number of cities (chunks)

    Returns:
    - List of integers of length n_cities
    """
    vec = np.array(vec, dtype=int)
    n_bits = int(np.log2(n_cities))
    if len(vec) != n_cities * n_bits:
        raise ValueError(f"Vector length should be {n_cities * n_bits}, got {len(vec)}")
    
    integers = []
    for i in range(n_cities):
        chunk = vec[i*n_bits:(i+1)*n_bits]
        # Convert binary chunk to integer
        val = 0
        for j, bit in enumerate(chunk):
            val += bit << (n_bits - j - 1)  # MSB first
        integers.append(val)
    
    return integers

@njit
def cost_tsp(x, adj_matrix, lam, gamma, mu):
    """
    Computes the TSP cost of a binary vector.

    Parameters:
    - x: Binary vector.
    - adj_matrix: Adjacency matrix.
    - lam: First coefficient in the cost function.
    - gamma: Second coefficient in the cost function.
    - mu: Third coefficient in the cost function.

    Returns:
    - TSP cost of a binary vector.
    """
    n = adj_matrix.shape[0]
    n_bits = math.ceil(math.log2(n))
    cities = decode_assignment(x, n, n_bits)

    tilde_x = int_to_bitstring(n - 1, n_bits)
    K_0 = [i for i in range(n_bits) if tilde_x[i] == 0]

    term1 = 0.0
    for t in range(n):
        term1 += H_valid(cities[t], tilde_x, K_0)

    term2 = 0.0
    for t in range(n):
        for tp in range(t + 1, n):
            term2 += H1(cities[t], cities[tp])

    term3 = 0.0
    for i in range(n):
        i_bin = int_to_bitstring(i, n_bits)
        for j in range(n):
            if i == j:
                continue
            j_bin = int_to_bitstring(j, n_bits)
            for t in range(n - 1):
                term3 += adj_matrix[i, j] * H1(cities[t], i_bin) * H1(cities[t + 1], j_bin)

    return lam * term1 + gamma * term2 + mu * term3

@njit(parallel=True)
def compute_all_costs(adj_matrix, lam, gamma, mu):
    """
    Computes the TSP cost of all 2^n binary vector.

    Parameters:
    - adj_matrix: Adjacency matrix.
    - lam: First coefficient in the cost function.
    - gamma: Second coefficient in the cost function.
    - mu: Third coefficient in the cost function.

    Returns:
    - Array of TSP costs of all 2^n binary vector.
    """
    n = adj_matrix.shape[0]
    n_bits = math.ceil(math.log2(n))
    total_bits = n * n_bits
    total_assignments = 2 ** total_bits

    costs = np.zeros(total_assignments)
    for idx in prange(total_assignments):
        x = int_to_bitstring(idx, total_bits)
        costs[idx] = cost_tsp(x, adj_matrix, lam, gamma, mu)

    return costs