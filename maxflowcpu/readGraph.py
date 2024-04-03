""" import numpy as np

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
np.savetxt(output_file_path, adjacency_matrix_int, fmt='%d')"""


import numpy as np

def mtx_to_csr(mtx_file_path):
    with open(mtx_file_path, 'r') as f:
        lines = f.readlines()

    # Filter out comments and empty lines
    lines = [line.strip() for line in lines if not line.startswith('%') and line.strip()]

    # Process the header to get matrix dimensions
    num_rows, num_cols, _ = map(int, lines[0].split())

    # Initialize lists to store matrix data
    rows, cols, data = [], [], []

    # Extract row, column indices, and data values
    for line in lines[1:]:
        row, col, value = line.split()
        rows.append(int(row) - 1)  # Adjust for 0-based indexing
        cols.append(int(col) - 1)  # Adjust for 0-based indexing
        data.append(float(value))

    # Convert lists to numpy arrays
    rows = np.array(rows)
    cols = np.array(cols)
    data = np.array(data)

    # Sort by rows, then columns to ensure CSR format
    sorted_indices = np.lexsort((cols, rows))
    rows = rows[sorted_indices]
    cols = cols[sorted_indices]
    data = data[sorted_indices]

    # Create CSR row pointer array
    row_ptr = np.zeros(num_rows + 1, dtype=int)
    np.add.at(row_ptr, rows + 1, 1)
    np.cumsum(row_ptr, out=row_ptr)

    return data, cols, row_ptr

def save_to_txt(data, cols, row_ptr, prefix):
    np.savetxt(f'{prefix}_data.txt', data, fmt='%f')
    np.savetxt(f'{prefix}_col_indices.txt', cols, fmt='%d')
    np.savetxt(f'{prefix}_row_ptr.txt', row_ptr, fmt='%d')

# Example usage
mtx_file_path = 'data/cage3.mtx'  # Update this to your file path
prefix = 'output_csr'  # Prefix for output files

data, cols, row_ptr = mtx_to_csr(mtx_file_path)
save_to_txt(data, cols, row_ptr, prefix)
