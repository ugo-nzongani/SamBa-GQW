import numpy as np
from numba import njit, prange

'''Portfolio optimization implementation.

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
def portfolio_cost(x, lam, gamma, sigma, mu, k):
    """
    Computes the portfolio optimization cost of a binary vector.

    Parameters:
    - x: Binary vector.
    - lam: Risk appetite.
    - gamma: Penalty coefficient.
    - sigma: Covariance matrix.
    - mu: Expected return of assets.
    - k: Number of assets to select (Hamming weight of the feasible solutions).

    Returns:
    - Portfolio optimization cost of a binary vector.
    """
    n = x.shape[0]
    xf = x.astype(np.float64)

    # First term
    quad_term = 0.0
    for i in range(n):
        for j in range(n):
            quad_term += sigma[i, j] * xf[i] * xf[j]
    quad_term *= lam

    # Second term
    linear_term = -np.dot(mu, xf)

    # Third term
    sum_x = np.sum(xf)
    penalty_term = gamma * (sum_x - k) ** 2

    return quad_term + linear_term + penalty_term

'''
@njit
def bitstring(i, n):
    x = np.empty(n, dtype=np.uint8)
    for b in range(n):
        x[n - 1 - b] = (i >> b) & 1
    return x
'''

@njit(parallel=True)
def compute_all_costs(n, lam, gamma, sigma, mu, k):
    """
    Computes the portfolio optimization cost of all 2^n binary vectors.

    Parameters:
    - n: Total number of assets (number of qubits).
    - lam: Risk appetite.
    - gamma: Penalty coefficient.
    - sigma: Covariance matrix.
    - mu: Expected return of assets.
    - k: Number of assets to select (Hamming weight of the feasible solutions).

    Returns:
    - Portfolio optimization cost of all 2^n binary vectors.
    """
    num_assignments = 1 << n
    costs = np.empty(num_assignments, dtype=np.float64)
    for i in prange(num_assignments):
        x = int_to_bitstring(i, n)
        costs[i] = portfolio_cost(x, lam, gamma, sigma, mu, k)
    return costs