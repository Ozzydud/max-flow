import networkx as nx
import random
from scipy.io import mmwrite
from scipy.sparse import csr_matrix

# Create an empty undirected graph
G = nx.Graph()

# Add 10,000 nodes
G.add_nodes_from(range(10000))

# Function to generate a random weight
def generate_weight():
    return random.uniform(0.1, 10.0)  # You can adjust the range as needed

# Add 500,000 random edges with weights
while G.number_of_edges() < 1000000:
    # Select two random nodes
    u = random.randint(0, 9999)
    v = random.randint(0, 9999)
    
    # Add an edge if it does not already exist and is not a self-loop
    if u != v and not G.has_edge(u, v):
        weight = generate_weight()
        G.add_edge(u, v, weight=weight)

print(f"The graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")

# Define source and sink
source = 0  # First node
sink = 9999  # Last node

# Check if there is a path from source to sink
if nx.has_path(G, source, sink):
    print("There is a path from the source to the sink.")
    matrix = nx.to_scipy_sparse_matrix(G, nodelist=sorted(G.nodes()), weight='weight', dtype=float, format='csr')
    # Save the matrix to a .mtx file
    mmwrite("output_graph.mtx", matrix)
else:
    print("No path from the source to the sink.")


