from collections import defaultdict
from scipy.interpolate import interp1d
import random
from typing import List, Set
import numpy as np
import scipy.sparse as sp
from itertools import combinations
from tqdm import tqdm
from functools import partial
import numbers
import itertools

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

def remove_duplicate_values(d):
    """
    Removes duplicate values in a dictionary based on hashable float representations.
    """
    seen = set()
    result = {}
    for key, value in d.items():
        val = value.item() if hasattr(value, "item") else float(value)
        if val not in seen:
            seen.add(val)
            result[key] = val
    return result

def hamming_neighbors(bitvector):
    """
    Computes the neighbors of a hypercube vertex.

    Parameters:
    - bitvector: Numpy array of 0s and 1s representing a solution.

    Returns:
    - List of numpy arrays representing the neighbors of `bitvector` according to hypercube-mixer connectivity.
    """
    neighbors = []
    n = bitvector.size
    for i in range(n):
        neighbor = bitvector.copy()
        neighbor[i] = 1 - neighbor[i]
        neighbors.append(bitstring_to_int(neighbor))
    return neighbors

def hamming_distance(v1, v2):
    """
    Computes the Hamming distance between two bitvectors.

    Parameters:
    - v1: First numpy array of 0s and 1s.
    - v2: Second numpy array of 0s and 1s.

    Returns:
    - Hamming distance between `v1` and `v2`.
    """
    return np.sum(v1 != v2)

def hamming_weight(v):
    """
    Computes the Hamming weight of a binary vector.
    (i.e., the number of 1s in the vector)

    Parameters:
    - v: Numpy array of 0s and 1s.

    Returns:
    - Hamming weight of `v` (integer).
    """
    return np.sum(v)

# XY-mixers

def xy_ring_neighbors(bitvec: np.ndarray) -> list:
    """
    Given a bit vector representing a state (np.array of 0s and 1s),
    return list of neighbors as new bit vectors reachable by XY swaps on ring.
    """
    n = len(bitvec)
    neighbors = []
    
    for i in range(n):
        j = (i + 1) % n
        if bitvec[i] != bitvec[j]:
            new_bitvec = bitvec.copy()
            new_bitvec[i], new_bitvec[j] = new_bitvec[j], new_bitvec[i]  # Swap bits
            neighbors.append(bitstring_to_int(new_bitvec))
    
    return neighbors

def uniform_superposition_hamming_range(n, k1, k2):
    """
    Returns a normalized numpy array of size 2**n representing a uniform superposition
    over all states with Hamming weight between k1 and k2 (inclusive).

    Parameters:
        n (int): Number of qubits.
        k1 (int): Minimum Hamming weight.
        k2 (int): Maximum Hamming weight.

    Returns:
        np.ndarray: Complex array of shape (2**n,) normalized.
    """
    if not (0 <= k1 <= k2 <= n):
        raise ValueError("k1 and k2 must satisfy 0 <= k1 <= k2 <= n.")

    dim = 2**n
    state = np.zeros(dim, dtype=np.complex128)

    for k in range(k1, k2 + 1):
        for positions in combinations(range(n), k):
            idx = sum(1 << pos for pos in positions)
            state[idx] = 1.0

    # Normalize
    norm = np.sqrt(np.sum(np.abs(state)**2))
    if norm > 0:
        state /= norm

    return state

'''
def sparse_diagonal_representation(matrix):
    rows, cols = matrix.shape
    sparse_dict = {}

    # Loop over all possible diagonal offsets
    for k in range(-rows + 1, cols):
        diag = np.diagonal(matrix, offset=k)  # explicitly using np.diagonal
        if diag[diag != 0].size > 0:
            sparse_dict[k] = diag

    return sparse_dict
'''

def hypercube_adjacency_sparse_final(n):
    """
    Final version: Return cleaned sparse diagonal representation of the hypercube adjacency matrix
    for dimension n. Ensures diagonals like -4 are correctly included.
    
    Output:
        A dictionary where:
            - Keys are diagonal offsets (±2^k),
            - Values are numpy arrays with 1s at valid edge positions (no all-zero arrays).
    """
    size = 1 << n  # Total number of vertices
    sparse = {}

    for k in range(n):
        offset = 1 << k
        # Positive diagonal (+offset)
        diag_pos = np.zeros(size - offset, dtype=int)
        for i in range(size - offset):
            j = i + offset
            if (i ^ j) == offset:
                diag_pos[i] = 1
        if np.any(diag_pos):
            sparse[offset] = diag_pos

        # Negative diagonal (-offset)
        diag_neg = np.zeros(size - offset, dtype=int)
        for j in range(size - offset):
            i = j + offset
            if (i ^ j) == offset:
                diag_neg[j] = 1
        if np.any(diag_neg):
            sparse[-offset] = diag_neg

    return sparse

class SamplingWalk:
    """
    SamplingWalk manages the initialization and manipulation of Hamiltonians
    for quantum sampling based on QAOA-like cost and mixer structure.

    Supports both QuTiP (NumPy, CPU) and Dynamiqs (JAX, GPU/TPU) backends.

    QuTip: https://qutip.readthedocs.io/en/latest/index.html
    Dynamiqs: https://www.dynamiqs.org/stable/
    """
    def __init__(self, n, cost, mixer, initial_state=None, use_density_matrix=False, local_mixer_gap=2,
                 use_qutip=True, convert_input_cost_fun=None,convert_input_neighbors_fun=None, cost_kwargs=None):
        """
        Parameters:
        - n: Size of the problem, i.e. number of binary variables.
        - cost: Array containing cost values of each solution or Function that computes the cost of a solution.
        - mixer: Array representing the connectivity of the mixer operator
        - initial_state: Quantum object of size 2^n representing the initial state of the system, if None it is set to uniform superposition
        - local_mixer_gap: Spectral gap of the local representation of the mixer
        - use_density_matrix: True if the states are to be represented with density matrices
        - use_qutip: True if the quantum simulation should be done with qutip, if False it is done with dynamiqs
        - convert_input_cost_fun: Function used to convert an integer to the correct input type of the cost function.
        - convert_input_neighbors_fun: Function used to convert an integer to the correct input type of the neighbors function 
        used in the sampling protocol.
        - cost_kwargs: Additional inputs of the cost function given as a dictionnary.
        """
        self.n_qubit = n
        self.mixer = mixer
        self.use_density_matrix = use_density_matrix
        self.local_mixer_gap = local_mixer_gap
        self.use_qutip = use_qutip
        self.initial_state = initial_state
        self.n_sample = 2 ** n
        self.sample = {}
        self.sampled_states = []
        self.collapse_op = []
        self.show_progress = False
        
        # Cost function input conversion
        if convert_input_cost_fun is None:
            self.convert_input_cost_fun = int_to_bitstring
        else:
            self.convert_input_cost_fun = convert_input_cost_fun
            
        # Cost function
        if callable(cost):
            self.cost_is_array = False
            base_cost = partial(cost, **cost_kwargs) if cost_kwargs else cost
            def wrapped_cost(x):
                if isinstance(x, numbers.Integral):
                    x = self.convert_input_cost_fun(x,self.n_qubit)
                return base_cost(x)
            self.cost = wrapped_cost
        else:
            self.cost = lambda x: cost[x]
            self.cost_is_array = True
            self.cost_array = cost
            
        # Neighbor function
        if convert_input_neighbors_fun is None:
            self.convert_input_neighbors_fun = int_to_bitstring
        else:
            self.convert_input_neighbors_fun = convert_input_neighbors_fun
        
        if use_qutip:
            import numpy as np
            self.np = np
            self.dtype = np.float64
        else:
            global jax
            import jax
            import jax.numpy as jnp
            #jax.config.update("jax_enable_x64", True)
            self.np = jnp
            self.dtype = jnp.float32

        self.hopping = self.np.ones(2 ** n, dtype=self.dtype)
        self.t_list = self.np.linspace(0, 1, 2 ** n, dtype=self.dtype)

        if use_qutip:
            self.qutip_initialization()
        else:
            self.dynamiqs_initialization()

        self.first_energies = []
        self.first_mean_gaps = []
        self.energies = []
        self.mean_gaps = []
        self.dt = 1.
        self.gap_dt = 1
        
        # Hamiltonian evolution options
        self.store_states = True
        self.store_final_state = True
        self.n_steps = 10000
        
    def qutip_initialization(self):
        """
        Initialization of quantum objects with QuTip
        """
        from qutip import Qobj, basis, ket2dm, QobjEvo, sesolve, mesolve
        globals().update({"Qobj": Qobj, "basis": basis, "ket2dm": ket2dm, "QobjEvo": QobjEvo, "sesolve": sesolve, "mesolve": mesolve})

        if self.initial_state is None:
            vec = sum(basis(2 ** self.n_qubit, i) for i in range(2 ** self.n_qubit))
            self.initial_state = ket2dm(vec).unit() if self.use_density_matrix else vec.unit()
        else:
            self.initial_state = self.initial_state

        if not self.cost_is_array:
            cost_list = [self.cost(self.convert_input_cost_fun(x,self.n_qubit)) for x in range(2 ** self.n_qubit)]
            self.hc = Qobj(self.np.diag(self.np.array(cost_list, dtype=self.dtype)))
        else:
            self.hc = Qobj(self.np.diag(self.np.array(self.cost_array, dtype=self.dtype)))

        self.hd = Qobj(-self.mixer)
        self.hamiltonian = QobjEvo([[self.hd, self.hopping]], tlist=self.t_list) + self.hc

    def dynamiqs_initialization(self):
        """
        Initialization of quantum objects with Dynamiqs
        """
        import dynamiqs as dq
        globals()['dq'] = dq

        dim = 2 ** self.n_qubit
        if self.initial_state is None:
            #basis_matrix = self.np.eye(dim, dtype=self.np.complex64)
            #vec = self.np.sum(basis_matrix, axis=0).reshape((-1, 1))
            vec = self.np.full((dim, 1), 1 / self.np.sqrt(dim), dtype=self.np.complex64)
            self.initial_state = dq.unit(dq.todm(vec)) if self.use_density_matrix else dq.unit(vec)
        else:
            self.initial_state = self.initial_state

        if not self.cost_is_array:
            cost_list = self.np.array(
                [self.cost(self.convert_input_cost_fun(x,self.n_qubit)) for x in range(dim)],
                dtype=self.dtype
            )
            #self.hc = dq.asqarray(self.np.diag(cost_list),layout=dq.dia)
            self.hc = dq.sparsedia_from_dict({0:cost_list})
        else:
            #self.hc = dq.asqarray(self.np.diag(self.np.array(self.cost_array, dtype=self.dtype)),layout=dq.dia)
            #self.hc = self.np.diag(self.np.array(self.cost_array, dtype=self.dtype))
            self.hc = dq.sparsedia_from_dict({0:self.cost_array})

        #self.hd = self.np.array(-self.mixer) #dq.asqarray()
        self.hd = -self.mixer #dq.sparsedia_from_dict(sparse_diagonal_representation(-self.mixer))
        self.update_hamiltonian(self.t_list,self.hopping,self.hd,self.hc)

    def sample_without_conflicts(self, symmetry_fun=lambda x: []):
        """
        Samples `self.n_sample` unique integers from [0, 2**self.n_qubit) such that no sample violates
        constraints defined by symmetry_fun(bitstring). Efficient for large `self.n_qubit`.
    
        Parameters:
        - symmetry_fun: Function returning a list of conflicting integers for a sampled integer.
                        The input type of `symmetry_fun` must be that of `self.cost`
    
        Returns:
        - List of `self.n_sample` unique integers satisfying conflict constraints.
        """
        sampled_inputs: Set[str] = set()
        forbidden_inputs: Set[str] = set()
        sampled_integers: List[int] = []
    
        while len(sampled_integers) < self.n_sample:
            candidate = random.getrandbits(self.n_qubit)
            correct_cost_input = self.convert_input_cost_fun(candidate,self.n_qubit)
    
            if candidate in sampled_inputs or candidate in forbidden_inputs:
                continue
    
            conflicts = symmetry_fun(correct_cost_input)
            if any(conflict in sampled_inputs for conflict in conflicts):
                continue
    
            sampled_inputs.add(candidate)
            sampled_integers.append(candidate)
            forbidden_inputs.update(conflicts)
    
        return sampled_integers
    
    def sampling_protocol(self, q, neighbors_fun, symmetry_fun=lambda x: []):
        """
        Executes the sampling protocol.
        
        Parameters:
        - q: Number of sampled solutions.
        - neighbors_fun: Function that returns all the neighbors of a given solution.
                        The neighbors of a solution MUST be of type int.
        - symmetry_fun: Function that returns all the symmetric (known equivalent) of a given solution.
        """
        self.sampled_states = []
        self.n_sample = min(q, 2 ** self.n_qubit)
        energy = {}
        sample = self.sample_without_conflicts(symmetry_fun)
        min_gap_list = []
        min_gap_energy = []
        for guess in sample:
            guess_correct_cost_input = self.convert_input_cost_fun(guess,self.n_qubit)
            neighbors = neighbors_fun(self.convert_input_neighbors_fun(guess,self.n_qubit))
            gap_list = [self.cost(guess) - self.cost(ngb) for ngb in neighbors]
            positive_gap_list = [x for x in gap_list if x > 0]
            self.sampled_states.append(guess)
            if positive_gap_list:
                max_gap = max(positive_gap_list)
                min_gap = min(positive_gap_list)
                #min_gap_list.append(min(positive_gap_list))
                #min_gap_energy.append(guess)
                energy[guess] = (self.cost(guess), max_gap, min_gap)
                #energy[guess] = (self.cost(guess), max_gap)
        #energy[self.np.argmin(min(min_gap_list))] = (self.cost(self.np.argmin(min(min_gap_list))),min(min_gap_list))
        self.sample = energy

    def compute_mean_gaps(self):
        """
        Computes the mean gaps for each unique energy level.
        """
        energy_gap_dict = defaultdict(list)
        for val in self.sample.values():
            energy_gap_dict[float(val[0])].append(val[1])

        energy_mean = {
            e: self.np.mean(self.np.array(gaps, dtype=self.dtype)) / self.local_mixer_gap
            for e, gaps in energy_gap_dict.items()
        }
        energy_mean = remove_duplicate_values(energy_mean)
        sorted_energies = sorted(energy_mean.keys())
        sorted_mean_gaps = sorted([energy_mean[e] for e in sorted_energies])[::-1]

        self.energies = sorted_energies
        self.mean_gaps = sorted_mean_gaps
        self.first_energies = self.np.array(sorted_energies)
        self.first_mean_gaps = self.np.array(sorted_mean_gaps)

    def update_hamiltonian(self, t_list, hopping, hd, hc):
        """
        Updates the Hamiltonian.
    
        Parameters:
        - t_list: Array of time values.
        - hopping: Hopping rate function represented as an array.
        - hd: Mixer Hamiltonian.
        - hc: Cost Hamiltonian
        """
        self.t_list = t_list
        self.hopping = hopping
        self.hd = hd
        self.hc = hc
        
        if self.use_qutip:
            self.hamiltonian = QobjEvo([[hd, hopping]], tlist=t_list) + hc
        else:
            
            def hopping_fun(t):
                # Use argmin of the absolute difference to get the "closest match"
                index = self.np.argmin(self.np.abs(self.t_list - t))
                return self.hopping[index]
            
            self.hamiltonian = dq.modulated(hopping_fun, self.hd) + self.hc
            #self.hamiltonian = dq.pwc(t_list, hopping, hd) + hc
        
    def interpolate(self, dt, gamma_list=[], annealing_time=True, t_max=1., stairs=False, cut=0., delta_min=0.,early_stop=0.,t_max_evolution=None):
        """
        Interpolates the hopping rate function.
    
        Parameters:
        - dt: Number of values for the interpolation.
        - gamma_list: Array containing the values gamma_k to be added to the hopping rate.
        - annealing_time: If True, the annealing time condition is respected.
        - t_max: Maximum time evolution, only counts if `annealing_time` is set to False.
        - stairs: True if the interpolation if piece-wise constant
        - cut: Minimum value allowed for the hopping rate
        - delta_min: Minimum gap difference allowed for the hopping rate
        """
        # If mean_gaps contains only one value, we must manually add values in the hopping rate for the interpolation
        if len(self.mean_gaps) == 1:
            raise ValueError("Not enough values to interpolate: add elements to gamma_list to solve this issue.")

        self.mean_gaps = self.first_mean_gaps[self.first_mean_gaps > cut]
        self.energies = self.first_energies[:len(self.mean_gaps)] 
        
        # We make sure that gamma_list is sorted in increasing order
        '''
        gamma_list = sorted(gamma_list)[::-1]
        
        gamma_list = [min(self.mean_gaps)] + gamma_list
        for i, gamma_k in enumerate(gamma_list[1:], 1):
            if gamma_k < gamma_list[i - 1] and gamma_k > cut:
                if self.use_qutip:
                    #self.mean_gaps.insert(-1,gamma_k) #self.mean_gaps.append(gamma_k)
                    #self.energies.insert(0, min(self.energies)-1)
                    self.mean_gaps = self.np.concatenate([self.mean_gaps,self.np.array([gamma_k])])
                    new_value = self.np.min(self.energies) - 1
                    self.energies = self.np.concatenate([self.np.array([new_value]), self.energies])
                else:
                    self.mean_gaps = self.np.concatenate([self.mean_gaps,self.np.array([gamma_k])])
                    new_value = self.np.min(self.energies) - 1
                    self.energies = self.np.concatenate([self.np.array([new_value]), self.energies])
        selected_gaps = [self.mean_gaps[0]]
        selected_energies = [self.energies[0]]
        for i in range(1,len(self.mean_gaps)):
            if self.np.abs(self.mean_gaps[i]-selected_gaps[-1]) >= delta_min:
                selected_gaps.append(self.mean_gaps[i])
                selected_energies.append(self.energies[i])

        if len(selected_gaps) > 1:
            self.mean_gaps = self.np.array(selected_gaps)
            self.energies = self.np.array(selected_energies)
        '''
        if annealing_time:
            annealing_time_list = []
            t = 0
            last_annealing_index = 0
            index_to_remove = []
            for i, gap in enumerate(self.mean_gaps):
                t_gap = self.np.pi / (2 * self.np.sqrt(2) * gap)
                if t_max_evolution is not None:
                    if t + t_gap <= t_max_evolution:
                        annealing_time_list.append(t+t_gap)
                        #print('Correct, gap:',gap,' Time:',t+t_gap,' tgap:',t_gap,' t:',t)
                        t += t_gap
                    elif t + gap <= t_max_evolution:
                        annealing_time_list.append(t+gap)
                        #print('Incorrect, gap:',gap,' Time:',t+gap)
                        t += gap
                    else:
                        index_to_remove.append(i)
                else:
                    annealing_time_list.append(t+t_gap)
                    t += t_gap
                '''
                if gap >= early_stop:
                    s += self.np.pi / (2 * self.np.sqrt(2) * gap)
                    annealing_time_list.append(s)
                    last_annealing_index = i
                else:
                    print('gap:',gap)
                    s += gap
                    annealing_time_list.append(s)
                '''

            if len(index_to_remove) > 0:
                self.mean_gaps = self.np.delete(self.mean_gaps,self.np.array(index_to_remove))
            #print(len(annealing_time_list),len(self.mean_gaps))
            #self.gap_dt = self.np.array(annealing_time_list)
            self.gap_dt = self.np.concatenate([self.np.array([0]),self.np.array(annealing_time_list)])
            '''
            if early_stop == 0:
                self.gap_dt = self.np.concatenate([self.np.array(annealing_time_list),self.np.array([s+annealing_time_list[last_annealing_index]])])
            else:
                self.gap_dt = self.np.concatenate([self.np.array(annealing_time_list),self.np.array([s+self.mean_gaps[-1]])])
            '''
            #self.gap_dt = self.np.concatenate([self.np.array(annealing_time_list),self.np.array([t+1e-2])])
            self.mean_gaps = self.np.concatenate([self.mean_gaps,self.np.array([0])])
            #print(len(self.gap_dt),len(self.mean_gaps))
        else:
            self.gap_dt = self.np.linspace(0, t_max, len(self.mean_gaps), dtype=self.dtype)
        
        if stairs:
            def stairs_interp(x, xp, fp):
                x = self.np.asarray(x,dtype=self.dtype)
                xp = self.np.asarray(xp,dtype=self.dtype)
                fp = self.np.asarray(fp,dtype=self.dtype)

                indices = self.np.searchsorted(xp, x, side='right') - 1
                indices = self.np.clip(indices, 0, len(fp) - 1)
                return self.np.take(fp, indices)

            # Define t_list as before
            t_list = self.np.linspace(self.gap_dt[0], self.gap_dt[-1], dt, dtype=self.dtype)

            # Compute hopping_interpolated using the stairs interpolation
            hopping_interpolated = stairs_interp(t_list, self.gap_dt, self.mean_gaps)
        else:
            interp_func = interp1d(self.gap_dt, self.mean_gaps, kind='linear', fill_value="extrapolate")
            t_list = self.np.linspace(self.gap_dt[0], self.gap_dt[-1], dt, dtype=self.dtype)
            hopping_interpolated = interp_func(t_list)

        self.dt = dt
        self.update_hamiltonian(t_list, self.np.array(hopping_interpolated), self.hd, self.hc)

    def evolve_qutip(self):
        """
        Hamiltonian evolution with QuTip.
        """
        options = {
            'store_states': self.store_states,
            'store_final_state': self.store_final_state,
            'nsteps': self.n_steps,
        }
        if self.show_progress:
            options['progress_bar'] = 'tqdm'
            
        if self.use_density_matrix:
            result = mesolve(self.hamiltonian, self.initial_state, self.t_list, self.collapse_op, e_ops=[self.hc], options=options)
        else:
            result = sesolve(self.hamiltonian, self.initial_state, self.t_list, e_ops=[self.hc], options=options)
        return result

    def evolve_dynamiqs(self):
        """
        Hamiltonian evolution with Dynamiqs
        """
        options = dq.Options(
            save_states = self.store_states
        )
        if self.use_density_matrix:
            result = dq.mesolve(self.hamiltonian, self.collapse_op, self.initial_state, self.t_list, exp_ops=[self.hc], options=options)
        else:
            method = dq.method.Tsit5(rtol= 1e-8,
                                    atol= 1e-8,
                                    safety_factor= 0.9,
                                    min_factor= 0.2,
                                    max_factor= 5.0,
                                    max_steps= 10000000,
                                    )
            result = dq.sesolve(self.hamiltonian, self.initial_state, self.t_list, exp_ops=[self.hc], options=options, method=method)
        return result
        
    def evolve(self):
        """
        Hamiltonian evolution
        """
        if self.use_qutip:
            return self.evolve_qutip()
        else:
            return self.evolve_dynamiqs()

    def settings(self,device='cpu',precision='single',show_progress=False):
        """
        Modifies the settings of QuTip/Dynamiqs.
    
        Parameters:
        - device: Device on which the evolution is computed, 'cpu', 'gpu' or 'tpu' (only for Dynamiqs).
        - precision: Floating point precision, 'single' (float32 & complex68) or 'double' (float64 & complex128) (only for Dynamiqs).
        - show_progress: If True, display the progress of the computation.
        """
        if self.use_qutip:
            self.qutip_settings(show_progress)
        else:
            self.dynamiqs_settings(device,precision,show_progress)

    def qutip_settings(self,show_progress=False):
        """
        Modifies the settings of QuTip.
    
        Parameters:
        - show_progress: If True, display the progress of the computation.
        """
        self.show_progress = show_progress
        
    def dynamiqs_settings(self,device='cpu',precision='single',show_progress=False):
        """
        Modifies the settings of Dynamiqs.
    
        Parameters:
        - device: Device on which the evolution is computed, 'cpu', 'gpu' or 'tpu'.
        - precision: Floating point precision, 'single' (float32 & complex68) or 'double' (float64 & complex128).
        - show_progress: If True, display the progress of the computation.
        """
        dq.set_device(device)
        dq.set_precision(precision)
        dq.set_progress_meter(progress_meter=show_progress)
        self.show_progress = show_progress