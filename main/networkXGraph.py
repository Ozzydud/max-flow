import networkx as nx
import random
from scipy.io import mmwrite
from scipy.sparse import csr_matrix
import sys

# Handling command-line arguments
if len(sys.argv) != 3:
    print("Usage: python script.py <number_of_nodes> <number_of_edges>")
    sys.exit(1)

number_of_nodes = int(sys.argv[1])
number_of_edges = int(sys.argv[2])

# Create an empty undirected graph
G = nx.Graph()

# Add X nodes
G.add_nodes_from(range(number_of_nodes))

# Function to generate a random weight
def generate_weight():
    return round(random.uniform(0.1, 1.0), 8)  # You can adjust the range as needed

# Add X random edges with weights
while G.number_of_edges() < number_of_edges:
    # Select two random nodes
    u = random.randint(0, number_of_nodes - 1)
    v = random.randint(0, number_of_nodes - 1)
    
    # Add an edge if it does not already exist and is not a self-loop
    if u != v and not G.has_edge(u, v):
        weight = generate_weight()
        G.add_edge(u, v, weight=weight)

print(f"The graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")

# Define source and sink
source = 0  # First node
sink = number_of_nodes - 1  # Last node

# Create a custom function to write the matrix
def write_matrix(G, filename):
    with open(filename, 'w') as file:
        n = G.number_of_nodes()
        edges = G.number_of_edges()
        file.write(f"%%MatrixMarket matrix coordinate real symmetric\n%\n{n} {n} {edges}\n")
        for u, v, data in G.edges(data=True):
            file.write(f"{u+1} {v+1} {data['weight']:.8f}\n")  # Adjust format here


# Check if there is a path from source to sink
if nx.has_path(G, source, sink):
    print("There is a path from the source to the sink.")
    #matrix = nx.to_scipy_sparse_array(G, nodelist=sorted(G.nodes()), weight='weight', dtype=float, format='csr')
    # Save the matrix to a .mtx file
    #mmwrite("output_graph.mtx", matrix)
    write_matrix(G, "custom_output_graph_test.mtx")
else:
    print("No path from the source to the sink.")


