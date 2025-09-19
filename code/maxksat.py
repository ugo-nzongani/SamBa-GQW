import numpy as np
import math
from numba import njit, prange
from numba.typed import List
from typing import List as PyList, Union

'''
MAX-k-SAT implementation.

numpy==2.2.5
numba==0.61.2
'''

def generate_uniform_kSAT(n, m, k, seed=None, unique=True):
    """
    Efficient generator for a random k-SAT instance with m clauses.

    Parameters:
    - n: Number of variables.
    - m: Number of clauses.
    - k: Number of literals per clause.
    - seed: Random seed for reproducibility.
    - unique: If True, ensures clauses are unique (slower).

    Returns:
    - List of k-literal clauses, each as a tuple.
    """
    if k < 1 or k > n:
        raise ValueError("The number k of literals per clause must respect 0 < k <= n")
    if unique and m > (math.comb(n, k) * 2**k):
        raise ValueError("Cannot generate more unique clauses than possible combinations.")

    if seed is not None:
        np.random.seed(seed)

    if unique:
        clauses = set()
        while len(clauses) < m:
            vars_ = np.random.choice(np.arange(1, n + 1), size=k, replace=False)
            signs = np.random.choice([-1, 1], size=k)
            clause = tuple(sign * var for var, sign in zip(vars_, signs))
            clauses.add(clause)
        return list(clauses)
    else:
        clauses = []
        for _ in range(m):
            vars_ = np.random.choice(np.arange(1, n + 1), size=k, replace=False)
            signs = np.random.choice([-1, 1], size=k)
            clause = tuple(sign * var for var, sign in zip(vars_, signs))
            clauses.append(clause)
        return clauses

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
def evaluate_maxk_sat(x, clauses):
    """
    Computes the cost of assignment x for a given Max-k-SAT instance.
    
    Parameters:
    - x: NumPy array of 0s and 1s representing the assignment.
    - clauses: List of clauses, where each clause is a tuple of integers.
    
    Returns:
    - Number of satisfied clauses.
    """
    cost = 0
    for clause in clauses:
        for lit in clause:
            idx = abs(lit) - 1
            if (lit > 0 and x[idx] == 1) or (lit < 0 and x[idx] == 0):
                cost += 1
                break
    return -cost

@njit(parallel=True)
def evaluate_all_maxksat(n, clauses):
    """
    Evaluates all 2^n binary assignments for Max-k-SAT clauses.

    Parameters:
    - n: Number of variables.
    - clauses: List of clauses, where each clause is a tuple of integers.
    
    Returns:
    - NumPy array of costs (negated satisfied clause counts).
    """
    num_assignments = 2 ** n
    results = np.empty(num_assignments, dtype=np.int32)
    for i in prange(num_assignments):
        x = int_to_bitstring(i, n)
        results[i] = evaluate_maxk_sat(x, clauses)
    return results

def convert_clauses_to_numba(clauses: PyList[Union[PyList[int], np.ndarray]]) -> List:
    """
    Converts a list of clauses into a Numba-typed list of NumPy arrays.

    Parameters:
    - clauses: List of clauses, where each clause is a tuple of integers.
    
    Returns:
    - ??.
    """
    typed_clauses = List()
    for clause in clauses:
        typed_clauses.append(np.array(clause, dtype=np.int32))
    return typed_clauses

def pretty_print_clauses(clauses, var_prefix='x', return_str=False):
    """
    Prints or returns a human-readable version of k-SAT clauses.

    Parameters:
    - clauses: List of clauses, each as a tuple of signed ints.
    - var_prefix: Prefix to use for variable names (default 'x').
    - return_str: If True, returns the pretty-printed clause string.

    Returns:
    - If return_str=True, returns the pretty-printed clause string.
    """
    lines = []
    for clause in clauses:
        literals = []
        for lit in clause:
            var = f"{var_prefix}{abs(lit)}"
            if lit < 0:
                literals.append(f"¬{var}")
            else:
                literals.append(var)
        clause_str = "(" + " ∨ ".join(literals) + ")"
        lines.append(clause_str)

    result = "\n".join(lines)
    if return_str:
        return result
    else:
        print(result)

# Values extracted from Table 10 of https://quantum-journal.org/papers/q-2019-07-18-167/pdf/

alpha_k = {
    3: 4.27,
    4: 9.93,
    5: 21.12,
    6: 43.37,
    7: 87.79,
    8: 176.54,
    9: 354.01,
    10: 708.92,
    11: 1418.71,
    12: 2838.28,
    13: 5677.41,
    14: 11355.67,
    15: 22712.20,
    }