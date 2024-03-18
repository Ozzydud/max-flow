import numpy as np

# Function to parse the .mtx file
def parse_mtx(mtx_lines):
    header_skipped = False
    matrix_data = []
    
    for line in mtx_lines:
        if line.startswith('%'):
            continue  # Skip comment lines
        elif not header_skipped:
            # First non-comment line should contain the matrix dimensions
            dimensions = line.strip().split()
            num_rows = int(dimensions[0])
            num_cols = int(dimensions[1])
            non_zero_entries = int(dimensions[2])
            header_skipped = True
        else:
            # Remaining lines will contain the matrix data
            i, j, val = line.strip().split()
            matrix_data.append((int(i), int(j), float(val)))
    
    return num_rows, num_cols, non_zero_entries, matrix_data

# Read the .mtx file
mtx_file_path = 'data/gre_1107.mtx'
with open(mtx_file_path, 'r') as file:
    lines = file.readlines()

# Parse the .mtx file
num_rows, num_cols, non_zero_entries, matrix_data = parse_mtx(lines)

# Initialize an adjacency matrix with zeros
adjacency_matrix = np.zeros((num_rows, num_cols))

# Populate the adjacency matrix with the values from the .mtx data
for i, j, val in matrix_data:
    # Subtract 1 from i and j to convert to 0-based index
    adjacency_matrix[i-1, j-1] = val

# Convert to integers, if needed
adjacency_matrix_int = np.rint(adjacency_matrix * 1000).astype(int)

# Save the adjacency matrix to a file
output_file_path = 'data/1107data.mtx' #Change to your path
np.savetxt(output_file_path, adjacency_matrix_int, fmt='%d')


