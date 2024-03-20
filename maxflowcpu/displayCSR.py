import numpy as np

def read_and_print_csr_components(csr_row_ptr_path, col_indices_path, data_path):
    # Load the CSR components from the .npy files
    csr_row_ptr = np.load(csr_row_ptr_path)
    col_indices = np.load(col_indices_path)
    data = np.load(data_path)

    # Print the contents of the CSR components
    print("CSR Row Pointers:", csr_row_ptr)
    print("Column Indices:", col_indices)
    print("Data:", data)

# Example usage
csr_row_ptr_path = 'data/csrgre_1107.mtx_csr_row_ptr.npy'
col_indices_path = 'data/csrgre_1107.mtx_col_indices.npy'
data_path = 'data/csrgre_1107.mtx_data.npy'

read_and_print_csr_components(csr_row_ptr_path, col_indices_path, data_path)
