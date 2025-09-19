import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from itertools import combinations
from collections import defaultdict
import dynamiqs as dq
import scipy.sparse as sp
from qutip import Qobj

def cost_quality_by_hamming(cost_values, k, n_bits=None):
    """
    Build dictionary mapping indices -> quality values in [0,1],
    keeping only indices with Hamming weight k.
    Indices with Hamming weight != k get value -1.

    Parameters:
    - cost_values: 1D numpy array of real numbers
    - k: target Hamming weight
    - n_bits: number of bits to represent indices (default: log2(len(cost_values)))

    Returns:
    - dict: {index: quality value in [0,1] or -1}
    """
    size = len(cost_values)
    if n_bits is None:
        n_bits = int(np.log2(size))
        if 2**n_bits != size:
            raise ValueError("Length of cost_values must be a power of 2 if n_bits is not given.")

    def hamming_weight(x):
        return bin(x).count("1")

    cost_quality_dict = {}
    selected = [(i, cost_values[i]) for i in range(size) if hamming_weight(i) == k]

    if selected:
        values = np.array([val for _, val in selected])
        vmin, vmax = values.min(), values.max()
        if vmax == vmin:
            norm = {i: 1.0 for i, _ in selected}  # all equal → all "best"
        else:
            # flip normalization: smaller = better (closer to 1)
            norm = {i: (vmax - val) / (vmax - vmin) for i, val in selected}
    else:
        norm = {}

    for i in range(size):
        cost_quality_dict[i] = norm.get(i, -1.0)  # -1 for non-selected indices

    return cost_quality_dict

def kron_n(*ops):
    """Kronecker product of multiple operators."""
    result = ops[0]
    for op in ops[1:]:
        result = np.kron(result, op)
    return result

def xy_mixer_matrix(n: int) -> np.ndarray:
    """
    Constructs the XY mixer Hamiltonian matrix with ring connectivity:
    H = 1/2 * sum_i (X_i X_{i+1} + Y_i Y_{i+1})
    """
    # --- Pauli matrices ---
    I = np.eye(2)
    X = np.array([[0, 1], [1, 0]])
    Y = np.array([[0, -1j], [1j, 0]])

    H = np.zeros((2**n, 2**n), dtype=complex)
    for i in range(n):
        j = (i + 1) % n
        ops_x = [I] * n
        ops_y = [I] * n
        ops_x[i], ops_x[j] = X, X
        ops_y[i], ops_y[j] = Y, Y
        H += (kron_n(*ops_x) + kron_n(*ops_y)) / 2
    return H

# --- Graph Utilities ---

def fixed_hamming_weight_states(n: int, h: int) -> list:
    """Returns all bitstrings of length n with Hamming weight h."""
    states = []
    for ones in combinations(range(n), h):
        bits = ['0'] * n
        for i in ones:
            bits[i] = '1'
        states.append(''.join(bits))
    return states

def ring_connected_hamming_graph(n: int, h: int) -> nx.Graph:
    """Graph where nodes are hamming-weight-h states, edges are allowed XY swaps."""
    states = fixed_hamming_weight_states(n, h)
    G = nx.Graph()
    G.add_nodes_from(states)

    for u in states:
        u_list = list(u)
        for i in range(n):
            j = (i + 1) % n
            if u_list[i] == '0' and u_list[j] == '1':
                v_list = u_list.copy()
                v_list[i], v_list[j] = v_list[j], v_list[i]
                v = ''.join(v_list)
                if v in G:
                    G.add_edge(u, v)
    return G

def annotate_edges_with_swap(G: nx.Graph, n: int) -> nx.Graph:
    """Annotate edges with the qubit swap index."""
    for u, v in G.edges():
        diffs = [i for i in range(n) if u[i] != v[i]]
        if len(diffs) == 2:
            i, j = sorted(diffs)
            if (j == (i + 1) % n) or (i == (j + 1) % n):
                G.edges[u, v]['swap'] = f"{i}<->{j}"
    return G

# --- Visualization ---

def generate_all_hamming_subgraphs(n: int):
    """Returns a dictionary of graphs for each Hamming weight."""
    graphs = {}
    for h in range(n + 1):
        G = ring_connected_hamming_graph(n, h)
        annotate_edges_with_swap(G, n)
        graphs[h] = G
    return graphs

def plot_all_subgraphs_grouped(graphs: dict, n: int, cost_quality_dict: dict, node_size=50, cmap="coolwarm"):
    """
    Plot all subgraphs, grouping those with the same number of nodes side by side.
    Node colors are determined by `cost_quality_dict`.
    """

    # Group graphs by number of nodes
    grouped = defaultdict(list)
    for h, G in graphs.items():
        grouped[len(G.nodes())].append((h, G))

    # Sort groups by number of nodes
    groups_sorted = sorted(grouped.items(), key=lambda x: x[0])

    # Grid dimensions
    num_rows = len(groups_sorted)
    max_cols = max(len(v) for _, v in groups_sorted)

    # Reduce figure size per subplot
    fig, axes = plt.subplots(
        num_rows, max_cols,
        figsize=(3 * max_cols, 2 * num_rows),
        constrained_layout=True
    )

    if num_rows == 1:
        axes = [axes]
    axes = np.array(axes).reshape(num_rows, max_cols)

    # Keep track of first axis for colorbar
    first_ax = None

    for row_idx, (num_nodes, group) in enumerate(groups_sorted):
        for col_idx, (h, G) in enumerate(group):
            ax = axes[row_idx, col_idx]
            pos = nx.spring_layout(G, seed=42, k=0.3)  # smaller spacing

            # Build node colors
            node_colors = [cost_quality_dict.get(int(node, 2), -1) for node in G.nodes()]

            nx.draw_networkx_nodes(
                G, pos, ax=ax, node_size=node_size,
                node_color=node_colors, cmap=plt.get_cmap(cmap),
                vmin=-1, vmax=1
            )
            nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.5)
            #nx.draw_networkx_labels(G, pos, ax=ax, font_size=6)
            ax.axis("off")

            if first_ax is None:
                first_ax = ax

        # Hide unused subplot slots
        for col_idx in range(len(group), max_cols):
            axes[row_idx, col_idx].axis("off")

    # Add a single large horizontal colorbar at the bottom
    sm = plt.cm.ScalarMappable(cmap=plt.get_cmap(cmap), norm=plt.Normalize(vmin=-1, vmax=1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, orientation='horizontal', fraction=0.05, pad=0.05)
    cbar.set_label("Quality", fontsize=10)

    plt.show()

def plot_all_subgraphs_sampled(graphs: dict, n: int, sampled_nodes: list, node_size=50):
    """
    Plot all subgraphs, grouping those with the same number of nodes side by side.
    Nodes in `sampled_nodes` are colored red, others gray.
    A legend indicates the meaning of colors.
    """

    # Group graphs by number of nodes
    grouped = defaultdict(list)
    for h, G in graphs.items():
        grouped[len(G.nodes())].append((h, G))

    # Sort groups by number of nodes
    groups_sorted = sorted(grouped.items(), key=lambda x: x[0])

    # Grid dimensions
    num_rows = len(groups_sorted)
    max_cols = max(len(v) for _, v in groups_sorted)

    # Reduce figure size per subplot
    fig, axes = plt.subplots(
        num_rows, max_cols,
        figsize=(3 * max_cols, 2 * num_rows),
        constrained_layout=True
    )

    if num_rows == 1:
        axes = [axes]
    axes = np.array(axes).reshape(num_rows, max_cols)

    for row_idx, (num_nodes, group) in enumerate(groups_sorted):
        for col_idx, (h, G) in enumerate(group):
            ax = axes[row_idx, col_idx]
            pos = nx.spring_layout(G, seed=42, k=0.3)  # smaller spacing

            # Build node colors: red if sampled, gray otherwise
            node_colors = ['red' if int(node, 2) in sampled_nodes else 'skyblue' for node in G.nodes()]

            nx.draw_networkx_nodes(G, pos, ax=ax, node_size=node_size, node_color=node_colors, alpha=0.7)
            nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.7)
            # nx.draw_networkx_labels(G, pos, ax=ax, font_size=6)
            ax.axis("off")
            #ax.set_title(f"Hamming {h} ({num_nodes} nodes})", fontsize=8)

        # Hide unused subplot slots
        for col_idx in range(len(group), max_cols):
            axes[row_idx, col_idx].axis("off")

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='red', label='Sampled nodes'),
                       Patch(facecolor='skyblue', label='Unsampled nodes')]
    fig.legend(handles=legend_elements, loc='upper center', ncol=1, fontsize=10)

    plt.show()

def xy_mixer_ring(n):
    """
    Generates the XY-mixer Hamiltonian with ring connectivity as a sparse matrix,
    using sparse Kronecker products.
    
    H_XY = -0.5 * sum_{(i,j) ∈ A_XY} (σ^x_i σ^x_j + σ^y_i σ^y_j)
    
    Parameters:
        n (int): Number of qubits.
    
    Returns:
        scipy.sparse.csr_matrix: The XY-mixer Hamiltonian matrix (sparse).
    """
    dim = 2 ** n
    H = sp.csr_matrix((dim, dim), dtype=np.complex128)
    
    # Pauli X and Y sparse matrices
    sx = sp.csr_matrix(np.array([[0, 1], [1, 0]], dtype=np.complex128))
    sy = sp.csr_matrix(np.array([[0, -1j], [1j, 0]], dtype=np.complex128))
    id2 = sp.identity(2, format='csr', dtype=np.complex128)
    
    for i in range(n):
        j = (i + 1) % n  # ring connectivity
        
        # Start with scalar 1 (identity in sparse format for kron)
        term_sx = sp.csr_matrix([[1]], dtype=np.complex128)
        term_sy = sp.csr_matrix([[1]], dtype=np.complex128)
        
        for k in range(n):
            if k == i:
                op_sx = sx
                op_sy = sy
            elif k == j:
                op_sx = sx
                op_sy = sy
            else:
                op_sx = id2
                op_sy = id2
            
            term_sx = sp.kron(term_sx, op_sx, format='csr')
            term_sy = sp.kron(term_sy, op_sy, format='csr')
        
        H -= 0.5 * (term_sx + term_sy)
    
    return H

def csr_to_offset_diagonal_dict(matrix):
    """
    Converts a scipy.sparse.csr_matrix into a dictionary with diagonal offsets as keys
    and lists of diagonal elements as values (including zeros), excluding fully zero diagonals.
    Optimized for large sparse matrices (e.g., 2**20 x 2**20).
    """
    rows, cols = matrix.shape
    matrix = matrix.tocoo()

    diagonals_data = {}
    diagonal_lengths = {}

    # Store only observed non-zero offsets
    seen_offsets = set()

    for row, col, val in zip(matrix.row, matrix.col, matrix.data):
        offset = col - row
        if offset not in diagonals_data:
            length = min(rows, cols, rows - abs(offset))
            diagonals_data[offset] = np.zeros(length, dtype=matrix.dtype)
            diagonal_lengths[offset] = length
            seen_offsets.add(offset)

        idx = row if offset >= 0 else col
        diagonals_data[offset][idx] = val

    # Build result dictionary only with observed offsets
    result = {
        offset: diagonals_data[offset].tolist()
        for offset in sorted(seen_offsets)
    }

    return result

def xy_ring_adjacency_sparse_final(n,use_qutip=True):
    mixer = -1*xy_mixer_ring(n)
    sparse_dq = csr_to_offset_diagonal_dict(mixer)
    sparse_dq = {
        int(offset): list(np.asarray(diagonal, dtype=np.complex64))
        for offset, diagonal in csr_to_offset_diagonal_dict(mixer).items()
    }
    if use_qutip:
        return mixer
    else:
        return dq.sparsedia_from_dict(sparse_dq)