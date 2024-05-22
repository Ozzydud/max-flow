import networkx as nx
import random
import sys

# Handling command-line arguments
if len(sys.argv) != 3:
    print("Usage: python script.py <number_of_nodes> <output_filename>")
    sys.exit(1)

number_of_nodes = int(sys.argv[1])
output_filename = sys.argv[2]

# Ensure the number of edges is exactly one less than the number of nodes for a star graph
number_of_edges = number_of_nodes - 1

# Create an empty directed graph
G = nx.DiGraph()

# Add nodes
G.add_nodes_from(range(number_of_nodes))

# Function to generate a random weight
def generate_weight():
    return round(random.uniform(0.1, 1.0), 8)

# Add edges from all nodes to the central node (sink)
for i in range(number_of_nodes - 1):
    G.add_edge(i, number_of_nodes - 1, weight=generate_weight())

print(f"The graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")

# Define source and sink
source = random.randint(0, number_of_nodes - 2)  # Randomly select one of the incoming nodes
sink = number_of_nodes - 1  # Last node is the central node (sink)

# Create a custom function to write the matrix
def write_matrix(G, filename):
    with open(filename, 'w') as file:
        n = G.number_of_nodes()
        edges = G.number_of_edges()
        file.write(f"%%MatrixMarket matrix coordinate real general\n%\n{n} {n} {edges}\n")
        
        # Collect edges sorted by source node index (u) and then by target node index (v)
        sorted_edges = sorted(G.edges(data=True), key=lambda x: (x[0], x[1]))
        
        # Write sorted edges to file
        for u, v, data in sorted_edges:
            file.write(f"{u+1} {v+1} {data['weight']:.8f}\n")  # Adjust format here

# Check if there is a path from source to sink
if nx.has_path(G, source, sink):
    print("There is a path from the source to the sink.")
    write_matrix(G, output_filename)
else:
    print("No path from the source to the sink.")
