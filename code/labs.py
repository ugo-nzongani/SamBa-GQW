import numpy as np
from numba import njit

'''LABS implementation.

numba==0.61.2
numpy==2.2.5
'''

@njit
def int_to_spin(i, n):
    """
    Converts an integer to a spin array.
    
    Parameters:
    - i: Integer.
    - n: Size of the spin array (minimal number of bits needed to write `i` in binary).
    
    Returns:
    - An array of spins.
    """
    if i > 0 and n < np.ceil(np.log2(i)):
        raise ValueError("Need larger n to encode the integer.")
    spin = np.empty(n, dtype=np.int8)
    for j in range(n):
        spin[n - j - 1] = 1 - 2 * ((i >> j) & 1)
    return spin

@njit
def spin_to_int(spin):
    """
    Converts a spin array to its integer representation.

    Parameters:
    - spin: Numpy array of spins (+1 or -1).

    Returns:
    - An integer corresponding to the binary encoding of the spin array.
    """
    result = 0
    n = spin.size
    for j in range(n):
        bit = (1 - spin[n - j - 1]) // 2
        result |= bit << j
    return result

@njit
def labs_energy(spin):
    """
    Computes the LABS energy of a spin.
    
    Parameters:
    - spin: Array of spins.
    
    Returns:
    - The LABS energy of a spin.
    """
    n = len(spin)
    e = 0
    for k in range(1, n):
        c = 0
        for i in range(n - k):
            c += spin[i] * spin[i + k]
        e += c * c
    return e
    
@njit
def labs_merit_factor(spin):
    """
    Computes the LABS merit factor of a spin.
    
    Parameters:
    - spin: Array of spins.
    
    Returns:
    - The LABS merit factor of a spin.
    """
    e = labs_energy(spin)
    if e == 0:
        merit = np.inf
    else:
        merit = n * n / (2 * e)
    return merit

@njit
def all_labs_energies(n):
    """
    Computes the LABS energies of all spin sequences of length `n`.
    
    Parameters:
    - n: Length of the spin sequences.
    
    Returns:
    - Array containing the LABS energies of all spin sequences of length `n`.
    """
    size = 2 ** n
    cost = np.empty(size)
    for i in range(size):
        spin = int_to_spin(i, n)
        cost[i] = labs_energy(spin)
    return cost

@njit
def all_labs_merit_factors(n):
    """
    Computes the LABS merit factor of all spin sequences of length `n`.
    
    Parameters:
    - n: Length of the spin sequences.
    
    Returns:
    - Array containing the LABS merit factor of all spin sequences of length `n`.
    """
    size = 2 ** n
    cost = np.empty(size)
    for i in range(size):
        spin = int_to_spin(i, n)
        e = labs_energy(spin)
        if e == 0:
            cost[i] = np.inf
        else:
            cost[i] = n * n / (2 * e)
    return cost