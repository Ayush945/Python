import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

def manual_pagerank(adj_matrix, damping_factor=0.85, max_iterations=100, tolerance=1.0e-6):
    """
    Calculates PageRank using Linear Algebra (Power Iteration) from scratch.
    """
    N = len(adj_matrix)
    
    # 1. Create the Transition Matrix (M)
    # We normalize the columns so they sum to 1 (Probability Distribution)
    M = np.zeros((N, N))
    for j in range(N):
        column_sum = sum(adj_matrix[:, j])
        if column_sum == 0:
            # Handle "Dangling Nodes" (pages with no outgoing links)
            M[:, j] = 1.0 / N
        else:
            M[:, j] = adj_matrix[:, j] / column_sum

    # 2. Create the "Google Matrix" (G) handling the Damping Factor
    # This accounts for the random chance that a user types a URL instead of clicking a link
    E = np.ones((N, N)) / N  # Matrix of all 1/N
    G = (damping_factor * M) + ((1 - damping_factor) * E)

    # 3. Power Iteration (The Optimization Loop)
    # Start with equal probability for all pages
    v = np.ones(N) / N 
    
    for i in range(max_iterations):
        v_new = np.dot(G, v)  # Matrix Multiplication: The core of Linear Algebra
        
        # Check for Convergence (Did the numbers stop changing?)
        if np.linalg.norm(v_new - v) < tolerance:
            print(f"Converged after {i} iterations.")
            return v_new
        
        v = v_new

    return v

# --- EXECUTION ---

# Define the Graph (0 points to 1, 1 points to 2, etc.)
# Let's say: 0=A, 1=B, 2=C, 3=D
# A links to B and C
# B links to C
# C links to A
# D links to C
edges = [
    (0, 1), (0, 2), 
    (1, 2), 
    (2, 0),
    (3, 2)
]
num_nodes = 4

# Build Adjacency Matrix (Rows = To, Cols = From)
A = np.zeros((num_nodes, num_nodes))
for start, end in edges:
    A[end][start] = 1 

print("Adjacency Matrix (Structure):")
print(A)
print("-" * 20)

# Run our manual algorithm
ranks = manual_pagerank(A)

print("Final PageRank Scores:")
for i, r in enumerate(ranks):
    print(f"Node {i}: {r:.4f}")

# --- VISUALIZATION---
G_vis = nx.DiGraph()
G_vis.add_edges_from(edges)
pos = nx.spring_layout(G_vis)

# Draw nodes sized by their PageRank score
node_sizes = [ranks[i] * 5000 for i in range(num_nodes)]
nx.draw(G_vis, pos, with_labels=True, node_size=node_sizes, node_color='lightblue', edge_color='gray', arrowsize=20)
plt.title("PageRank Visualization (Size = Importance)")
plt.show()