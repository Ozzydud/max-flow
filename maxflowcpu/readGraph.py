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
np.savetxt(output_file_path, adjacency_matrix_int, fmt='%d')


 """
import numpy as np

def mtx_to_csr_and_save(mtx_file_path, output_prefix):
    with open(mtx_file_path, 'r') as f:
        lines = f.readlines()

    # Filter out comments and empty lines
    lines = [line.strip() for line in lines if not line.startswith('%') and line.strip()]

    # Process the header
    num_rows, num_cols, num_entries = map(int, lines[0].split())

    # Initialize arrays
    rows = []
    cols = []
    data = []

    # Extracting data from the lines
    for entry in lines[1:]:
        row, col, value = map(float, entry.split())
        rows.append(int(row) - 1)  # Adjust for 0-based indexing
        cols.append(int(col) - 1)  # Adjust for 0-based indexing
        data.append(value)

    # Convert to CSR format
    row_counts = np.bincount(rows, minlength=num_rows)
    csr_row_ptr = np.zeros(num_rows + 1, dtype=np.int32)
    np.cumsum(row_counts, out=csr_row_ptr[1:])
    col_indices = np.array(cols, dtype=np.int32)
    data = np.array(data, dtype=np.float32)

    # Saving CSR components to files
    np.save(f'{output_prefix}_csr_row_ptr.npy', csr_row_ptr)
    np.save(f'{output_prefix}_col_indices.npy', col_indices)
    np.save(f'{output_prefix}_data.npy', data)

# Usage example
mtx_file_path = 'data/cage3.mtx'
output_prefix = 'data/csrgre_1107.mtx'
mtx_to_csr_and_save(mtx_file_path, output_prefix)

def npy_to_txt(npy_file_path, txt_file_path):
    array = np.load(npy_file_path)
    np.savetxt(txt_file_path, array, fmt='%f')

csr_row_ptr_path = 'data/csrgre_1107.mtx_csr_row_ptr.npy'
col_indices_path = 'data/csrgre_1107.mtx_col_indices.npy'
data_path = 'data/csrgre_1107.mtx_data.npy'

# Convert each CSR component
npy_to_txt('data/csrgre_1107.mtx_csr_row_ptr.npy', 'data/csr_row_ptr.txt')
npy_to_txt('data/csrgre_1107.mtx_col_indices.npy', 'data/col_indices.txt')
npy_to_txt('data/csrgre_1107.mtx_data.npy', 'data/data.txt')