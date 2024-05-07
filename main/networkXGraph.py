import networkx as nx
import random
from scipy.io import mmwrite
from scipy.sparse import csr_matrix
import sys

# Handling command-line arguments
if len(sys.argv) != 4:
    print("Usage: python script.py <number_of_nodes> <number_of_edges> <output_filename>")
    sys.exit(1)

number_of_nodes = int(sys.argv[1])
number_of_edges = int(sys.argv[2])
output_filename = sys.argv[3]

# Create an empty undirected graph
G = nx.Graph()

# Add X nodes
G.add_nodes_from(range(number_of_nodes))

# Function to generate a random weight
def generate_weight():
    return round(random.uniform(0.1, 1.0), 8)  # You can adjust the range as needed

# Add X random edges with weights
max_possible_edges = (number_of_nodes * (number_of_nodes - 1)) // 2  # Maximum possible edges in an undirected graph
while G.number_of_edges() < number_of_edges and G.number_of_edges() < max_possible_edges:
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
        
        # Collect edges sorted by column index (v)
        sorted_edges_by_col = sorted(G.edges(data=True), key=lambda x: (x[1], x[0]))
        
        # Write sorted edges to file
        for u, v, data in sorted_edges_by_col:
            file.write(f"{u+1} {v+1} {data['weight']:.8f}\n")  # Adjust format here

# Check if there is a path from source to sink
if nx.has_path(G, source, sink):
    print("There is a path from the source to the sink.")
    write_matrix(G, output_filename)
else:
    print("No path from the source to the sink.")
