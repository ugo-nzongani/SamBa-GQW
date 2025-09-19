from numba import njit, prange
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

'''Maximum Independent Set implementation.

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

def generate_triangular_lattice(M, spacing=1.0):
    """
    generate a triangular lattice with a roughly square layout containing approximately M points.
    
    Parameters:
        M (int): Number of points (approximate due to triangular structure)
        spacing (float): Distance between neighboring points
    
    Returns:
        np.ndarray: Array of shape (N, 2) with x, y coordinates of the points
    """
    side_length = int(np.sqrt(M))  # Estimate number of points per row/column
    points = []
    for row in range(side_length):
        for col in range(side_length):
            x = col * spacing + (row % 2) * (spacing / 2)
            y = row * (spacing * np.sqrt(3) / 2)
            points.append((x, y))
    return np.array(points)

def select_closest_points(points, N, p):
    """
    Select the L = N/p closest points to the center.
    
    Parameters:
        points (np.ndarray): Array of shape (N, 2) containing the lattice points.
        N (int): Number of desired points.
        p (float): Ratio parameter (0 < p <= 1).
    
    Returns:
        np.ndarray: Array of selected closest points.
    """
    L = int(N / p)
    center = np.mean(points, axis=0)
    distances = np.linalg.norm(points - center, axis=1)
    closest_indices = np.argsort(distances)[:L]
    return points[closest_indices]

def sample_points(selected_points, N):
    """Randomly sample N points from the selected L points."""
    indices = np.random.choice(len(selected_points), N, replace=False)
    return selected_points[indices]

def build_graph(sampled_points, threshold=1.1):
    """Builds a graph from sampled points where edges exist between neighbors."""
    g = nx.Graph()
    for i, p1 in enumerate(sampled_points):
        g.add_node(i, pos=tuple(p1))
        for j, p2 in enumerate(sampled_points[:i]):
            if np.linalg.norm(p1 - p2) < threshold:
                g.add_edge(i, j)
    return g

def rewire_graph(g, rewire_prob):
    """Rewires edges in the graph with a given probability."""
    edges = list(g.edges())
    nodes = list(g.nodes())
    for edge in edges:
        if np.random.rand() < rewire_prob:
            g.remove_edge(*edge)
            new_target = np.random.choice(nodes)
            while new_target == edge[0] or g.has_edge(edge[0], new_target):
                new_target = np.random.choice(nodes)
            g.add_edge(edge[0], new_target)
    return g

def compute_node_weights(g):
    """Computes the weight of each node based on its degree."""
    num_nodes = len(g.nodes)
    degrees = {node: g.degree(node) / (num_nodes + 1) for node in g.nodes}
    max_C = max(degrees.values()) if degrees else 1  # Avoid division by zero
    weights = [1 + (C/max_C)*10 for node, C in degrees.items()]
    return weights

def plot_lattice_with_graph(points, selected_points, sampled_points, g):
    """Plots the triangular lattice with selected and sampled points linked by edges."""
    plt.figure(figsize=(8, 8))
    plt.scatter(points[:, 0], points[:, 1], c='blue', edgecolors='k', label='All Points')
    
    # Compute bounding box for selected points
    min_x, min_y = np.min(selected_points, axis=0) - 0.1
    max_x, max_y = np.max(selected_points, axis=0) + 0.1
    plt.gca().add_patch(plt.Rectangle((min_x, min_y), max_x - min_x, max_y - min_y, fill=False, edgecolor='red', lw=1.5))
    
    # get node positions and weights for plotting
    pos = nx.get_node_attributes(g, 'pos')
    weights = nx.get_node_attributes(g, 'weight')
    node_sizes = [weights[node] for node in g.nodes]
    
    nx.draw(g, pos, with_labels=True, node_color='green', edge_color='black', node_size=node_sizes, cmap=plt.cm.viridis)
    
    plt.axis('equal')
    plt.legend()
    plt.title("Triangular Lattice with graph Connections")
    plt.show()

def plot_fixed_graph(g,show_labels=False,save=False,figname='ud_graph.png'):
    """Plots the graph with fixed node positions."""
    pos = nx.get_node_attributes(g, 'pos')  # Retrieve stored positions
    
    #plt.figure(figsize=(8, 8))
    nx.draw(g, pos, with_labels=show_labels, node_color='lime', edge_color='black', node_size=300)
    if save:
        plt.savefig(figname,dpi=200)
    plt.show()

def max_weight_sum_on_edges(weights, adj_matrix):
    """
    Efficiently computes the maximum w_i + w_j over all edges (i,j) in an undirected graph.

    Parameters:
    - weights: 1D numpy array of vertex weights (shape: [n])
    - adj_matrix: 2D numpy array of shape [n, n], symmetric, with 1s indicating edges

    Returns:
    - max_sum: maximum value of w_i + w_j over all edges
    """
    weights = np.asarray(weights)
    # Get indices of the upper triangle where there is an edge
    i_indices, j_indices = np.triu_indices_from(adj_matrix, k=1)
    edge_mask = adj_matrix[i_indices, j_indices] > 0
    i_edges = i_indices[edge_mask]
    j_edges = j_indices[edge_mask]

    if len(i_edges) == 0:
        return 0  # No edges

    sum_weights = weights[i_edges] + weights[j_edges]
    return np.max(sum_weights)
'''

def max_weight_sum_on_edges(weights, adj_matrix):
    n = len(weights)
    total = 0
    for i in range(n):
        for j in range(i + 1, n):
            if adj_matrix[i, j] == 1:
                total += weights[i] + weights[j]
    return total
'''

@njit
def mis_cost(x, weights, adj_matrix, penalty_coeff):
    """
    Computes the MIS cost of a binary vector.

    Parameters:
    - x: Binary vector.
    - weights: Array of weights for vertices.
    - adj_matrix: Adjacency matrix.
    - penalty_coeff: Penalty coefficient.

    Returns:
    - The MIS cost of a binary vector.
    """
    w_term = 0.0
    penalty = 0.0
    n = x.shape[0]
    for i in range(n):
        if x[i] == 1:
            w_term += weights[i]
            for j in range(i + 1, n):
                if adj_matrix[i, j] and x[j] == 1:
                    penalty += 1
    return -w_term + penalty_coeff * penalty

@njit(parallel=True)
def all_mis_costs(n, weights, adj_matrix, penalty_coeff):
    """
    Computes the MIS cost of all 2^n binary vectors.

    Parameters:
    - n: Size of the vector (number of qubits).
    - weights: Array of weights for vertices.
    - adj_matrix: Adjacency matrix.
    - penalty_coeff: Penalty coefficient.

    Returns:
    - Array containing the MIS cost of all 2^n binary vectors.
    """
    total_assignments = 1 << n  # 2**n
    costs = np.empty(total_assignments, dtype=np.float64)
    for i in prange(total_assignments):
        x = int_to_bitstring(i, n)
        costs[i] = mis_cost(x, weights, adj_matrix, penalty_coeff)
    return costs
